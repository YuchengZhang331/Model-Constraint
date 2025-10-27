from swemnics import solvers as Solvers
from swemnics.adcirc_problem import ADCIRCProblem
import numpy as np

def setup_problem():
    dt = 0.25
    alpha = 0.00014051891708
    t_f = 3 * 24 * 3600
    nt = int(t_f / dt)

    prob = ADCIRCProblem(
        adios_file="/work/09633/yzhang331/frontera/Small_Inlet_Case/ADCIRC2FENICS_Example/testout1",
        spherical=False,
        solution_var="flux",
        friction_law="quadratic",
        wd=False,
        wd_alpha=alpha,
        dt=dt,
        nt=nt,
        dramp=0.0001,
    )
    prob.mag = 1.0
    solver = Solvers.DGExplicit(prob)
    
    return prob, solver

def evaluate_rhs(prob, solver, state: np.ndarray, time: float, mag: float) -> np.ndarray:
    prob.t = time
    solver.problem.t = time
    prob.mag = mag
    solver.init_weak_form()
    return solver.rhs_sol(state)
