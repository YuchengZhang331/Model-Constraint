import os
import pickle
import numpy as np
import jax
from jax import jit, grad, value_and_grad, random, vmap
from jax.example_libraries import stax, optimizers
import jax.numpy as jnp
import time
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"
from jax import config
config.update("jax_enable_x64", True)

init_random_params, predict = stax.serial(
    stax.Dense(8000), stax.Relu,
    stax.Dense(5000), stax.Relu,
    stax.Dense(8000), stax.Relu,
    stax.Dense(10026)
)

key = random.PRNGKey(0)
_, init_params = init_random_params(key, (-1, 10026))


with open('/scratch/09633/yzhang331/Small_Inlet_mag_train/data/model_params_epoch_16.pkl', 'rb') as f:
    loaded_params = pickle.load(f)
    
init_params = loaded_params    


import subprocess
index_mapping = np.load("/work/09633/yzhang331/frontera/Small_Inlet_Case/index_mapping_s_inlet.npy")
bathymetry = np.load("/work/09633/yzhang331/frontera/Small_Inlet_Case/inlet_bathymetry.npz")["bathymetry"].T.squeeze()
node_size = 632
bathy_array = np.zeros(632*3)
def nodal_normalize_to_physics_input(data_nnorm):
    data_ori = jnp.array(data_nnorm)
    
    water_level = data_ori[:node_size]
    water_depth = water_level + bathymetry
    data_ori = data_ori.at[:node_size].set(water_depth) 
    u_0 = data_ori[index_mapping].flatten()
    print("u_0[0]",u_0[0],u_0[3])
    return u_0

bathy_p_array = np.array(nodal_normalize_to_physics_input(bathy_array))
bathy_p_array = bathy_p_array.astype(np.float64)

def call_physics_solver(inputs, time_batch, mag_batch):
    # start_time_p = time.time()
    inputs = np.array(inputs)
    inputs[:,:] += bathy_p_array
    # Save inputs to temporary files
    np.save("/scratch/09633/yzhang331/tmp/inputs_2.npy", np.array(inputs))
    np.save("/scratch/09633/yzhang331/tmp/times_2.npy", np.array(time_batch))
    np.save("/scratch/09633/yzhang331/tmp/mag_2.npy", np.array(mag_batch))
    # Run the parallel physics script
    subprocess.run([
        "mpirun", "-n", "50",
        "python", "/work2/09633/yzhang331/frontera/Small-Inlet-Mag-Train/run_parallel_rhs_2.py"
    ], check=True)

    # Load result from file
    output = np.load("/scratch/09633/yzhang331/tmp/rhs_output_2.npy")
    return output


@jax.custom_jvp
def call_physics_solver_wrapper(inputs, time_batch, mag_batch):
    shape = jax.ShapeDtypeStruct((inputs.shape[0], 10026), jnp.float64)
    return jax.experimental.io_callback(call_physics_solver, shape, inputs, time_batch, mag_batch)

# Define the JVP rule (set to zero, no gradient information is used)
@call_physics_solver_wrapper.defjvp
def call_physics_solver_jvp(primals, tangents):
    t_value, physics_input = primals
    t_dot, physics_input_dot = tangents

    # Compute the primal output (normal function call)
    primal_out = call_physics_solver_wrapper(t_value, physics_input)

    # Zero gradient because JAX cannot differentiate through physics solvers
    tangent_out = jnp.zeros_like(primal_out)  

    return primal_out, tangent_out

left_bc_indices = np.load("/work2/09633/yzhang331/frontera/small_inlet_data_generate/left_bc_indices.npy")

left_bc_indices = left_bc_indices[::3]

# -----------------------------
# Loss and optimizer
# # -----------------------------
h_indices = np.arange(0, 10026, 9)[:, None] + np.array([0, 1, 2])
def loss_fn(params, inputs, mag_batch, time_batch, targets, batch_key):
    #add noise
    key, subkey = jax.random.split(batch_key)
    inputs = inputs - bathy_p_array
    noise = 0.001 * jax.random.normal(subkey, ())
    inputs_noise = inputs
    inputs_noise = inputs_noise.at[h_indices].add(noise)

    preds = vmap(predict, in_axes=(None, 0))(params, inputs)
    
    preds_noise = vmap(predict, in_axes=(None, 0))(params, inputs_noise)
    physics_noise = call_physics_solver_wrapper(inputs_noise, time_batch, mag_batch)
    
    return jnp.mean((preds_noise - physics_noise*6500) ** 2 + (preds - targets*6500) ** 2)

@jit
def train_step(params, inputs, mag_batch, time_batch, targets, batch_key, opt_state):
    loss, grads = value_and_grad(loss_fn)(params, inputs, mag_batch, time_batch, targets, batch_key)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, loss

# Choose optimizer
learning_rate = 1e-5
opt_init, opt_update, get_params = optimizers.adam(learning_rate)
opt_state = opt_init(init_params)

# -----------------------------
# Batch slicing function (no shuffling)
# -----------------------------
import threading
import queue
def slice_batches_npy(batch_size, N=1788570):
    data_dir = '/scratch/09633/yzhang331/Small-inlet-data-mags/'

    train_inputs = np.load(data_dir + 'u_train_shuffle.npy', mmap_mode='r')[:]
    train_targets = np.load(data_dir + 'du_train_shuffle.npy', mmap_mode='r')[:]
    mag_terms = np.load(data_dir + 'mag_shuffle.npy', mmap_mode='r')[:]
    time_array = np.load(data_dir + 'time_shuffle.npy', mmap_mode='r')[:]

    for i in range(0, N, batch_size):
        yield (
            np.array(train_inputs[i:i + batch_size]),
            np.array(mag_terms[i:i + batch_size]),
            np.array(train_targets[i:i + batch_size]),
            np.array(time_array[i:i + batch_size])
        )
# -----------------------------
# Training loop
# -----------------------------
num_epochs = 50000
batch_size = 317711
batch_size = 60000
losses = []
validation_losses = []

seed = 42
key = jax.random.PRNGKey(seed)
for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0
    # for inputs_batch, boundary_batch, targets_batch, time_batch in prefetch_batches(h5_path, batch_size):
    for inputs_batch, mag_batch, targets_batch, time_batch in slice_batches_npy(batch_size):    
        
        key, batch_key = jax.random.split(key)
        params, opt_state, batch_loss = train_step(
            get_params(opt_state),
            inputs_batch,
            mag_batch,
            time_batch,
            targets_batch, batch_key,
            opt_state
        )
        epoch_loss += batch_loss
        num_batches += 1
        print(num_batches)
        print(f"batch loss = {batch_loss}")
    epoch_loss /= num_batches
    losses.append(epoch_loss)
    
    if epoch % 1 == 0:
        print(f"Epoch {epoch + 1}, Train Loss: {epoch_loss:.6f}")

    if epoch % 1 == 0:
        with open(f'/scratch/09633/yzhang331/Small_Inlet_mag_train/mc/s_model_params_epoch_{epoch+1}.pkl', 'wb') as f:
            pickle.dump(get_params(opt_state), f)
        with open(f'/scratch/09633/yzhang331/Small_Inlet_mag_train/mc/s_losses_epoch_{epoch+1}.pkl', 'wb') as f:
            pickle.dump({"train_losses": losses}, f)
        print(f"Saved model + loss at epoch {epoch + 1}")