from mpi4py import MPI
import numpy as np
from rhs_solver import evaluate_rhs, setup_problem
import socket
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = socket.gethostname()

def run_parallel(state_array: np.ndarray, time_array: np.ndarray, mag_array: np.ndarray):
    n = comm.bcast(None if rank != 0 else state_array.shape[0], root=0)
    dof = comm.bcast(None if rank != 0 else state_array.shape[1], root=0)
    # print(f"1 [Rank {rank}] ", flush=True)
    local_n = n // size
    remainder = n % size
    start = rank * local_n + min(rank, remainder)
    end = start + local_n + (1 if rank < remainder else 0)
    local_len = end - start

    # Allocate local arrays
    local_states = np.empty((local_len, dof), dtype=np.float64)
    local_times = np.empty((local_len,), dtype=np.float64)
    
    # Scatter the states
    if rank == 0:
        counts = [(local_n + (r < remainder)) * dof for r in range(size)]
        displs = [sum(counts[:r]) for r in range(size)]
        flat_states = state_array.reshape(-1)
    else:
        counts = displs = flat_states = None
    # print(f"2 [Rank {rank}] ", flush=True)
    comm.Scatterv([flat_states, counts, displs, MPI.DOUBLE], local_states.reshape(-1), root=0)
    # print(f"3 [Rank {rank}] ", flush=True)
    # Scatter the times
    if rank == 0:
        time_counts = [local_n + (r < remainder) for r in range(size)]
        time_displs = [sum(time_counts[:r]) for r in range(size)]
    else:
        time_counts = time_displs = time_array = None

    comm.Scatterv([time_array, time_counts, time_displs, MPI.DOUBLE], local_times, root=0)

    # Scatter the mags
    if rank == 0:
        mag_counts = [local_n + (r < remainder) for r in range(size)]
        mag_displs = [sum(mag_counts[:r]) for r in range(size)]
    else:
        mag_counts = mag_displs = mag_array = None

    local_mags = np.empty((local_len,), dtype=np.float64)
    comm.Scatterv([mag_array, mag_counts, mag_displs, MPI.DOUBLE], local_mags, root=0)

    # Set up mesh once
    prob, solver = setup_problem()

    local_rhs_list = []
    for i, (s, t, m) in enumerate(zip(local_states, local_times, local_mags)):
        try:
            rhs = evaluate_rhs(prob, solver, s, t, m)
            local_rhs_list.append(rhs)
        except Exception as e:
            print(f"[Rank {rank}] ❌ Failed at time {t:.2f}, mag {m:.2f}: {e}", flush=True)
            local_rhs_list.append(np.full_like(s, np.nan))  # or raise error
    try:
        local_rhs = np.stack(local_rhs_list)
    except Exception as e:
        print(f"[Rank {rank}] ❌ Could not stack local_rhs_list: {e}", flush=True)
        local_rhs = np.full((local_len, dof), np.nan, dtype=np.float64)
    
    if rank == 0:
        rhs_all = np.empty((n, dof), dtype=np.float64)
    else:
        rhs_all = None

    # Gather result
    gather_counts = [(local_n + (r < remainder)) * dof for r in range(size)]
    gather_displs = [sum(gather_counts[:r]) for r in range(size)]
    if rank == 0:
        comm.Gatherv(
            sendbuf=local_rhs.reshape(-1),
            recvbuf=[rhs_all.reshape(-1), gather_counts, gather_displs, MPI.DOUBLE],
            root=0
        )
    else:
        comm.Gatherv(local_rhs.reshape(-1), None, root=0)
    return rhs_all if rank == 0 else None



if __name__ == "__main__":
    # print(f"[Rank {rank}] running on {host}", flush=True)
    if rank == 0:
        # state_array = np.load("/scratch/09633/yzhang331/tmp/inputs.npy", mmap_mode='r')
        # time_array = np.load("/scratch/09633/yzhang331/tmp/times.npy", mmap_mode='r')
        # state25 = np.load("/scratch/09633/yzhang331/Small_Inlet_MPI_Test/u_f_eulerdg025c.npy", mmap_mode='r')        # shape: (num_snapshots, dof + 1)

        # state_array = state25[15000:15000+5000, 1:] 
        state_array = np.load("/scratch/09633/yzhang331/tmp/inputs_2.npy", mmap_mode='r')
        # print("state_array.shape",state_array.shape)
        # time_array = np.arange(15001, 15001 + state_array.shape[0]) * 0.25
        time_array = np.load("/scratch/09633/yzhang331/tmp/times_2.npy", mmap_mode='r').squeeze()
        # print("time_array.shape",time_array.shape)
        mag_array = np.load("/scratch/09633/yzhang331/tmp/mag_2.npy", mmap_mode='r').squeeze()
    else:
        state_array = None
        time_array = None
        mag_array = None
    # Use run_parallel!
    result = run_parallel(state_array, time_array, mag_array)

    if rank == 0:
        np.save("/scratch/09633/yzhang331/tmp/rhs_output_2.npy", result)