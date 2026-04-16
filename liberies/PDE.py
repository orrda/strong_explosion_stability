from diffrax import diffeqsolve, ODETerm, PIDController, SaveAt, DiscreteTerminatingEvent, DirectAdjoint
from diffrax import Dopri5, Kvaerno5, Event, Euler, Heun, Tsit5, Dopri8, KenCarp5, Bosh3
import optimistix as optx
import jax.numpy as jnp
import jax

from typing import Any
import numpy as np

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

class PatchedDiscreteTerminatingEvent(DiscreteTerminatingEvent):
    root_finder: Any = None
@jax.jit
def delta(U, C):
	return C ** 2 - (1 - U) ** 2

@jax.jit
def delta1_xi(U, C, omega, delt, gamma):
    return U * (1 - U) * (1 - U - delt) - (C ** 2) * (3 * U +(- omega + 2 * delt)/gamma)

@jax.jit
def delta2_xi(U, C, omega, delt, gamma):
	return C * (1 - U) * (1 - U - delt) - (gamma - 1) * C * U * (2 - 2 * U + delt) / 2 - (C ** 3) + (2 * delt + (gamma - 1) * omega) * (C ** 3)/(2 * gamma * (1 - U))

@jax.jit
def ode_sys_by_xi(xi, UC, args):
    omega, delt, gamma = args
    U=UC[0]
    C=UC[1]

    dU_dx = delta1_xi(U, C, omega, delt, gamma)
    dC_dx = delta2_xi(U, C, omega, delt, gamma)

    return jnp.stack([dU_dx, dC_dx])/(delta(U, C) * xi)


def U_event(t, y, args):
    return 0.5 - (y[0] - 0.5)**2

def C_event(t, y, args):
    return y[1]

def sonic_event(t, y, args):
    return y[1]**2 - (1.0 - y[0])**2

import functools
@functools.partial(jax.jit, static_argnames=['stop_at_sonic', 'save_dense', 'max_steps'])
def solve_PDE(omega, delt, x_begin = 1, x_end = 0, gamma = 5/3, stop_at_sonic = False, save_dense=True, max_steps=10000):

    U_init = 2/(gamma + 1)
    C_init = jnp.sqrt(2 * gamma * (gamma - 1)) / (gamma + 1)

    if stop_at_sonic:
        cond_fn = lambda t, y, args, **kwargs: jnp.min(jnp.array([
            U_event(t, y, args), 
            C_event(t, y, args), 
            sonic_event(t, y, args)
        ]))
    else:
        cond_fn = lambda t, y, args, **kwargs: jnp.min(jnp.array([
            U_event(t, y, args), 
            C_event(t, y, args)
        ]))

    event = Event(cond_fn, optx.Bisection(rtol=1e-12, atol=1e-12))

    term = ODETerm(ode_sys_by_xi)
    solver = Dopri5()
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


if __name__ == "__main__":
    omega = 4.25
    delt = 0.25
    sol = solve_PDE(omega, delt, x_begin=1, x_end=0, gamma = 5/3, stop_at_sonic=True)
    last_xi = sol.ts[-1]

    xi_arr = jnp.linspace(last_xi, 1, 100)

    for xi in xi_arr:
        U = sol.evaluate(xi)[0]
        C = sol.evaluate(xi)[1]
        plt.plot(U, C, 'bo')
    line = jnp.linspace(0, 1, 100)
    plt.plot(line, 1-line, 'r--')
    plt.xlabel("U")
    plt.ylabel("C")
    plt.show()