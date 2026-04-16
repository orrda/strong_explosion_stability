import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp 
import optimistix as optx
import equinox as eqx

from diffrax import Event, ODETerm, PIDController, diffeqsolve, SaveAt, DirectAdjoint
from PDE import *
from solution import solution, solve_PDE
from type3matrix import *

jax.config.update("jax_enable_x64", True)

class Perturbation(eqx.Module):
    omega: float
    gamma: float
    q: float
    l: float
    sol: solution
    delt: float


    def __init__(self, omega, gamma, q, l):
        self.omega = omega
        self.gamma = gamma
        self.q = q
        self.l = l

        self.sol = solution(omega=self.omega, gamma=self.gamma)
        self.delt = self.sol.delt

    @jax.jit
    def A(self, xi):
        sol = self.sol
        inv_MM = jnp.linalg.inv(sol.MM(xi))
        NN = sol.NN(xi)
        NNq = sol.NNq(xi)
        NNl = sol.NNl(xi)

        A = inv_MM @ (NN + self.q * NNq + (self.l * (self.l + 1) * NNl))

        return A

        
    @jax.jit
    def boundary_condition(self):
        sol = self.sol
        q = self.q

        dG_boundery = - sol.omega * (sol.gamma + 1)/(sol.gamma - 1) - sol.dGdx(1.0)
        dUr_boundery = 2 * q / (sol.gamma + 1) - sol.dUdx(1.0)
        dUt_boundery = -2 / (sol.gamma + 1)
        dP_boundery = 2 * (2 * (q + 1) - sol.omega) / (sol.gamma + 1) - sol.dPdx(1.0)

        return jnp.array([dG_boundery, dUr_boundery, dUt_boundery, dP_boundery])

    @jax.jit
    def solve(self, P_thresh, epsilon):
        max_steps = 10000
        save_dense = True

        Y_1 = self.boundary_condition()
        
        vector_field = lambda xi, Y, args: self.A(xi) @ Y

        def event_cond_fn(t, y, args, **kwargs):
            return P_thresh - jnp.abs(y[3])

        event = Event(event_cond_fn)

        term = ODETerm(vector_field)
        solver = Dopri5()
        stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

        num_sol = diffeqsolve(
            term, 
            solver, 
            t0=1, 
            t1=self.sol.xi_final + epsilon, 
            dt0=None, 
            y0=Y_1, 
            args=(),
            stepsize_controller=stepsize_controller,
            saveat=SaveAt(dense=save_dense, t1=True),
            max_steps=max_steps,
            throw=False,
            event=event,
            adjoint=DirectAdjoint() 
        )
        return num_sol

    @jax.jit
    def P_val(self, P_thresh, epsilon):
        num_sol = self.solve(P_thresh, epsilon)
        P_last = num_sol.ys[-1, 3]
        return P_last
    
@jax.jit
def P_by_q(sol, q_arr, l):
    def compute_P(q):
        per = Perturbation(omega=sol.omega, gamma=sol.gamma, q=q, l=l)
        return per.P_val(P_thresh=1e+6, epsilon=1e-5)
    
    return jax.vmap(compute_P)(q_arr)
    

@jax.jit
def get_q(solution, l, q_guess = -0.5):

    def fn(q, args):
        per = Perturbation(omega=solution.omega, gamma=solution.gamma, q=q, l=l)
        return jnp.abs(per.P_val(P_thresh=1e+3, epsilon=1e-5))

    solver = optx.BFGS(rtol=1e-5, atol=1e-5)
    sol = optx.minimise(fn, solver, q_guess, args=(), max_steps=100)
    return sol.value




if __name__ == "__main__":
    gamma = 5/3
    omega = 0

    sol = solution(omega=omega, gamma=gamma)

    alpha = 1/(1-sol.delt)

    q_start = -2
    q_end = 2
    l_arr = jnp.linspace(0, 2, 200)
    q_arr = jnp.linspace(q_start, q_end, 200)


    s_arr = alpha * q_arr

    P_arr = jax.vmap(lambda l: P_by_q(sol, q_arr, l))(l_arr)
    P_arr = jnp.log(jnp.rot90(P_arr))


    plt.figure(figsize=(10, 6))

    plt.imshow(P_arr, aspect='auto', extent=[l_arr.min(), l_arr.max(), q_arr.min(), q_arr.max()], cmap='gray')


    plt.colorbar(label='P')
    plt.xlabel("l")
    plt.ylabel("s")
    plt.title("P vs s and l")
    plt.grid()
    plt.show()