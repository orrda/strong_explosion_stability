import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp 
import optimistix as optx
import equinox as eqx

from diffrax import Event, ODETerm, PIDController, diffeqsolve, SaveAt, DirectAdjoint
from PDE_log import *
from solution_log import solution, solve_PDE


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
    def A(self, t):
        sol = self.sol
        inv_MM = jnp.linalg.inv(sol.MM(t))
        NNr = sol.NNr(t)
        NNq = sol.NNq(t)
        NNl = sol.NNl(t)

        dxi_dt = delta0(sol.U(t), sol.C(t)) * sol.xi(t)

        A = inv_MM @ (NNr + self.q * NNq + self.l * (self.l + 1) * NNl) * dxi_dt

        return A

        
    @jax.jit
    def boundary_condition(self):
        sol = self.sol
        q = self.q

        dG_boundery = - sol.omega * (sol.gamma + 1)/(sol.gamma - 1) - sol.dGdxi(-0.)
        dUr_boundery = 2 * q / (sol.gamma + 1) - sol.dUdxi(-0.)
        dUt_boundery = -2 / (sol.gamma + 1)
        dP_boundery = 2 * (2 * (q + 1) - sol.omega) / (sol.gamma + 1) - sol.dPdxi(-0.)

        return jnp.array([dG_boundery, dUr_boundery, dUt_boundery, dP_boundery])

    @jax.jit
    def solve(self, P_thresh, epsilon):
        max_steps = 1000
        save_dense = True

        Y_1 = self.boundary_condition()
        
        vector_field = lambda t, Y, args: self.A(t) @ Y

        def event_cond_fn(t, y, args, **kwargs):
            return P_thresh - jnp.abs(y[3])

        event = Event(event_cond_fn)

        term = ODETerm(vector_field)
        solver = Dopri8()
        stepsize_controller = PIDController(rtol=1e-7, atol=1e-7)

        num_sol = diffeqsolve(
            term, 
            solver, 
            t0=0, 
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
        last_t = num_sol.ts[-1]
        P_last = num_sol.evaluate(last_t)[3]
        return P_last
    
    
@jax.jit
def P_by_q(sol, q_arr, l):
    def compute_P(q):
        per = Perturbation(omega=sol.omega, gamma=sol.gamma, q=q, l=l)
        return per.P_val(P_thresh=1e+12, epsilon=1e-5)
    
    return jax.vmap(compute_P)(q_arr)

@jax.jit
def q_by_l(sol, l):
    def func(q, args):
        per = Perturbation(omega=sol.omega, gamma=sol.gamma, q=q, l=l)
        return per.P_val(P_thresh=1e+12, epsilon=1e-5)

    solver = optx.Bisection(rtol=1e-7, atol=1e-7)
    root = optx.root_find(func, solver, y0=-0.5, throw=False, options=dict(lower=-1.0, upper=0.0))
    return root.value


if __name__ == "__main__":
    gamma = 5/3
    omega = 4.25
    delt = 0.25

    sol = solution(omega=omega, gamma=gamma, delt=delt)

    t_arr = jnp.linspace(0, 1, 1000) * sol.xi_final
    U_arr = sol.U(t_arr)
    C_arr = sol.C(t_arr)

    plt.plot(U_arr, C_arr, ".")
    plt.xlabel("U")
    plt.ylabel("C")
    plt.title("U vs C")
    plt.grid()
    plt.show()





    alpha = 1/(1-sol.delt)
    l_arr = jnp.linspace(0, 1, 150)
    s_arr = jnp.linspace(-1, 0, 150)
    q_arr = s_arr / alpha

    func = lambda l: P_by_q(sol, q_arr, l)

    P_arr = jax.vmap(func)(l_arr)

    Q, L = np.meshgrid(s_arr, l_arr, indexing='ij')

    P_arr = np.log(np.abs(P_arr))

    plt.figure()
    plt.pcolormesh(L, Q, P_arr, shading='auto', cmap='viridis')
    plt.colorbar(label='Difference Norm')
    plt.xlabel('l')
    plt.ylabel('q')
    plt.title('P_arr as a function of q and l')
    plt.show()