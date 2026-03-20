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
    def Y_0(self):
        sol = self.sol
        q = self.q

        dG_boundery = - sol.omega * (sol.gamma + 1)/(sol.gamma - 1) - sol.dGdxi(-0.)
        dUr_boundery = 2 * q / (sol.gamma + 1) - sol.dUdxi(-0.)
        dUt_boundery = -2 / (sol.gamma + 1)
        dP_boundery = 2 * (2 * (q + 1) - sol.omega) / (sol.gamma + 1) - sol.dPdxi(-0.)

        return jnp.array([dG_boundery, dUr_boundery, dUt_boundery, dP_boundery])

    @jax.jit
    def Y_end(self, t):
        sol = self.sol

        NNr = sol.NNr(t)
        NNq = sol.NNq(t)
        NNl = sol.NNl(t)

        NN = NNr + self.q * NNq + self.l * (self.l + 1) * NNl

        v1 = NN[0,:]
        v2 = NN[1,:]
        v4 = NN[3,:]

        first = - sol.gamma * sol.P(t) * v1
        second = sol.xi(t) * (1 - sol.U(t)) * sol.G(t) * v2
        third = - sol.P(t) * v4

        return first + second + third

    @jax.jit
    def resi(self, q, n):
        last_t = self.sol.xi_final
        grid_t = jnp.linspace(0, 1, 200) * last_t

        xi_arr = self.sol.xi(grid_t)
        dxi = jnp.diff(xi_arr)
        A_arr = self.A(grid_t[:-1])

        I_arr = jnp.tile(jnp.eye(4), (A_arr.shape[0], 1, 1))

        dYdt = I_arr + A_arr * dxi

        Y_1 = jax.lax.scan(lambda Y, dY: (Y + dY, None), self.Y_0(), dYdt)

        return Y_1 - self.Y_end(grid_t[-1])










@jax.jit
def resi_q(l, q, n):
    per = Perturbation(omega=4.25, gamma=5/3, q=q, l=l)
    return per.resi(q, n)


if __name__ == "__main__":
    gamma = 5/3
    omega = 3.2
    delt = 0

    sol = solution(omega=omega, gamma=gamma, delt=delt)

    alpha = 1/(1-sol.delt)
    l = 2
    s_real = jnp.linspace(-2, 2, 100)
    s_img = jnp.linspace(-2, 2, 100)
    s_arr = s_real[:, None] + 1j * s_img
    q_arr = s_arr / alpha

    # Use a double vmap since q_arr is a 2D array and we want 'q' to be a scalar in each evaluation
    res = jax.vmap(jax.vmap(lambda q: resi_q(l, q, 100)))(q_arr)
    res = jnp.log(jnp.linalg.norm(res) + 1e-12)  # Adding a small constant to avoid log(0)

    # To plot the magnitude, we can take the norm of the residual vector
    plt.imshow(res, extent=[s_img.min(), s_img.max(), s_real.min(), s_real.max()], aspect='auto', origin='lower')
    plt.colorbar()
    plt.xlabel('Imag(s)')
    plt.ylabel('Real(s)')
    plt.title(f'Residuals for l={l}')
    plt.show()
