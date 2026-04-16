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
def delta0(U, C):
	return C ** 2 - (1 - U) ** 2

@jax.jit
def delta1(U, C, omega, delt, gamma):
    return U * (1 - U) * (1 - U - delt) - (C ** 2) * (3 * U +(- omega + 2 * delt)/gamma)

@jax.jit
def delta2(U, C, omega, delt, gamma):
	return C * (1 - U) * (1 - U - delt) - (gamma - 1) * C * U * (2 - 2 * U + delt) / 2 - (C ** 3) + (2 * delt + (gamma - 1) * omega) * (C ** 3)/(2 * gamma * (1 - U))

@jax.jit
def ode_sys_by_t(t, y, args):
    omega, delt, gamma = args

    dU_dt = delta1(y[0], y[1], omega, delt, gamma)
    dC_dt = delta2(y[0], y[1], omega, delt, gamma)
    dXi_dt = delta0(y[0], y[1]) * y[2]


    return jnp.stack([dU_dt, dC_dt, dXi_dt])


def U_event(t, y, args):
    return 0.5 - (y[0] - 0.5)**2

def C_event(t, y, args):
    return y[1]

def sonic_event(t, y, args):
    return y[1]**2 - (1.0 - y[0])**2

import functools
@functools.partial(jax.jit, static_argnames=['stop_at_sonic', 'save_dense', 'max_steps'])
def solve_PDE(omega, delt, x_begin = 0, x_end = -1e+12, gamma = 5/3, stop_at_sonic = False, save_dense=True, max_steps=10000):

    U_init = 2/(gamma + 1)
    C_init = jnp.sqrt(2 * gamma * (gamma - 1)) / (gamma + 1)
    Xi_init = 1.0

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

    term = ODETerm(ode_sys_by_t)
    solver = Dopri8()
    stepsize_controller = PIDController(rtol=1e-9, atol=1e-9)


    num_sol = diffeqsolve(
        term, 
        solver, 
        t0=x_begin, 
        t1=x_end, 
        dt0=None, 
        y0=jnp.array([U_init, C_init, Xi_init]), 
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
    sol = solve_PDE(omega, delt, x_begin=0, x_end=-1e+12, gamma = 5/3, stop_at_sonic=True)
    last_t = sol.ts[-1]
    interval = jnp.logspace(-10, 0, 100)
    tau_arr = last_t * interval
    print(f"last tau: {last_t}")


    U_arr = [sol.evaluate(t)[0] for t in tau_arr]
    C_arr = [sol.evaluate(t)[1] for t in tau_arr]

    last_U = sol.evaluate(last_t)[0]
    last_C = sol.evaluate(last_t)[1]



    line = jnp.linspace(0, 1, 100)
    plt.plot(line, 1-line, 'r--')
    plt.plot(last_U, last_C, 'go', label='Last Point')
    plt.plot(U_arr, C_arr, '.')
    plt.xlabel("U")
    plt.ylabel("C")
    plt.show()