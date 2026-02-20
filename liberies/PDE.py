from diffrax import diffeqsolve, ODETerm, Dopri5, PIDController, SaveAt, DiscreteTerminatingEvent, DirectAdjoint, Kvaerno5
import jax.numpy as jnp
import jax
from typing import Any
import numpy as np

import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")

class PatchedDiscreteTerminatingEvent(DiscreteTerminatingEvent):
    root_finder: Any = None

def delta(U, C):
	return C ** 2 - (1 - U) ** 2

def delta1_xi(U, C, omega, delt, gamma):
    return U * (1 - U) * (1 - U - delt) - (C ** 2) * (3 * U +(- omega + 2 * delt)/gamma)

def delta2_xi(U, C, omega, delt, gamma):
	return C * (1 - U) * (1 - U - delt) - (gamma - 1) * C * U * (2 - 2 * U + delt) / 2 - (C ** 3) + (2 * delt + (gamma - 1) * omega) * (C ** 3)/(2 * gamma * (1 - U))


def ode_sys_by_xi(xi, UC, omega, delt, gamma):
    U=UC[0]
    C=UC[1]

    dU_dx = delta1_xi(U, C, omega, delt, gamma)/(delta(U, C) * xi)
    dC_dx = delta2_xi(U, C, omega, delt, gamma)/(delta(U, C) * xi)

    return jnp.stack([dU_dx, dC_dx])



def solve_PDE(omega, delt, x_begin = 1, x_end = 0, gamma = 5/3, stop_at_sonic = False, save_dense=True, max_steps=10000):

    U_init = 2/(gamma + 1)
    C_init = jnp.sqrt(2 * gamma * (gamma - 1)) / (gamma + 1)

    def vector_field(xi, UC, args):
        omega, delt, gamma = args
        return ode_sys_by_xi(xi, UC, omega, delt, gamma)

    def base_event(t, y, args, **kwargs):
        return (y[1] <= 0) | (y[0] <= 0) | (1 <= y[0])
    
    def sonic_event(t, y, args, **kwargs):
         return y[0] + y[1] - 1 <= 0


    if stop_at_sonic:
        func = lambda t, y, args, **kwargs: sonic_event(t, y, args, **kwargs) | base_event(t, y, args, **kwargs)
    else:
        func = base_event
    event = PatchedDiscreteTerminatingEvent(func)

    term = ODETerm(vector_field)
    solver = Kvaerno5()
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
    

    num_sol = diffeqsolve(
        term, 
        solver, 
        t0=x_begin, 
        t1=x_end, 
        dt0=None, 
        y0=jnp.array([U_init, C_init]), 
        args=(omega, delt, gamma),
        stepsize_controller=stepsize_controller,
        saveat=SaveAt(dense=save_dense, t1=True),
        max_steps=max_steps,
        throw=False,
        event=event,
        adjoint=DirectAdjoint() 
    )

    return num_sol