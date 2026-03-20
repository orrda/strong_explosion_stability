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

        dG_boundery, dUr_boundery, dUt_boundery, dP_boundery = jnp.broadcast_arrays(
            dG_boundery, dUr_boundery, dUt_boundery, dP_boundery
        )

        return jnp.array([dG_boundery, dUr_boundery, dUt_boundery, dP_boundery])

    @jax.jit
    def solve(self, P_thresh, epsilon):
        max_steps = 2000
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
        P_last = num_sol.evaluate(last_t)[0]
        return P_last
    
    
@jax.jit
def P_by_q(sol, q_arr, l):
    def compute_P(q):
        per = Perturbation(omega=sol.omega, gamma=sol.gamma, q=q, l=l)
        return per.P_val(P_thresh=1e+12, epsilon=1e-5)
    
    return jax.vmap(compute_P)(q_arr)
    

@jax.jit
def get_q(solution, l, q_guess = -0.5):

    def fn(q, args):
        per = Perturbation(omega=solution.omega, gamma=solution.gamma, q=q, l=l)
        return jnp.abs(per.P_val(P_thresh=1e+6, epsilon=1e-5))

    solver = optx.BFGS(rtol=1e-5, atol=1e-5)
    sol = optx.minimise(fn, solver, q_guess, args=(), max_steps=100)
    return sol.value

@jax.jit
def P_by_omega_analytic(omega):
    sol = solution(omega=omega, gamma=5/3)
    l = 1.0
    s = -0.4
    alpha = 1/(1-sol.delt)
    q = s / alpha

    per = Perturbation(omega=omega, gamma=5/3, q=q, l=l)
    return per.P_val(P_thresh=1e+12, epsilon=1e-8)

@jax.jit
def P_by_omega_q(omega, q):
    sol = solution(omega=omega, gamma=5/3)
    l = 1.0

    P_arr = P_by_q(sol, q_arr=q, l=l)
    return P_arr


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
    l = 1
    s = -0.4 + 0.1j
    q = s / alpha

    per = Perturbation(omega=omega, gamma=gamma, q=q, l=l)
    num_sol = per.solve(P_thresh=1e+12, epsilon=1e-9)
    last_t = num_sol.ts[-1]

    print(f"last per t: {last_t}, last t sol: {sol.xi_final}")

    xi_arr = sol.xi(t_arr)
    Y = jax.vmap(lambda t: num_sol.evaluate(t))(t_arr)

    d_G_arr = Y[:, 0]
    d_Ur_arr = Y[:, 1]
    d_Ut_arr = Y[:, 2]
    d_P_arr = Y[:, 3]

    Y_0 = per.boundary_condition()
    print(f"Boundary condition: {Y_0}")

    plt.plot([1,1,1,1], Y_0, "ro", label="Boundary Condition")
    plt.plot(xi_arr, d_G_arr, ".", label="dG")
    plt.plot(xi_arr, d_Ur_arr, ".", label="dUr")
    plt.plot(xi_arr, d_Ut_arr, ".", label="dUt")
    plt.plot(xi_arr, d_P_arr, ".", label="dP")
    plt.xlabel("xi")
    plt.ylabel("Perturbation")
    plt.title("Perturbations vs xi")
    plt.legend()
    plt.grid()
    plt.show()


    s_real = jnp.linspace(-1, 0, 100)
    s_img = jnp.linspace(0, 1, 100)
    s_arr = s_real[:, None] + 1j * s_img[None, :]
    q_arr = s_arr / alpha
    P_arr = jax.vmap(jax.vmap(lambda q: P_by_q(sol, jnp.array([q]), l)[0]))(q_arr)
    #P_arr = jnp.log(jnp.abs(P_arr))

    plt.imshow(jnp.log(jnp.abs(P_arr)), extent=(s_real[0], s_real[-1], s_img[0], s_img[-1]), aspect='auto', origin='lower')
    plt.colorbar(label="P")
    plt.xlabel("s (Real)")
    plt.ylabel("s (Imaginary)")
    plt.title("P vs s (Real and Imaginary)")
    plt.grid()
    plt.show()


    omega_arr = jnp.linspace(1., 4., 50)
    s_arr = jnp.linspace(-0.405, -0.395, 50)

    q_arr = s_arr / alpha

    P_arr = []
    for omega in omega_arr:
        P_arr.append(P_by_omega_q(omega, q_arr))
        print(f"Computed P for omega: {omega}")
    
    P_arr = jnp.array(P_arr)
    P_arr = jnp.log(jnp.abs(P_arr))
    P_arr = jnp.where(jnp.isinf(P_arr), 0, P_arr)

    plt.imshow(P_arr, extent=(omega_arr[0], omega_arr[-1], q_arr[0], q_arr[-1]), aspect='auto', origin='lower')
    plt.colorbar(label="P")
    plt.xlabel("Omega")
    plt.ylabel("q")
    plt.title("P vs Omega and q")
    plt.grid()
    plt.show()

    l_arr = jnp.linspace(0, 4, 200)
    s_arr = jnp.linspace(-1, 0, 200)

    q_arr = s_arr / alpha

    P_arr = jax.vmap(lambda l: P_by_q(sol, q_arr, l))(l_arr)

    plt.imshow(jnp.log(jnp.abs(P_arr)), extent=(s_arr[0], s_arr[-1], l_arr[0], l_arr[-1]), aspect='auto', origin='lower')
    plt.colorbar(label="P")
    plt.xlabel("s")
    plt.ylabel("l")
    plt.title("P vs s and l")
    plt.grid()
    plt.show()


