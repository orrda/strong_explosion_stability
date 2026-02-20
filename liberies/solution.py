import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp  # Added JAX import
import optimistix as optx
import equinox as eqx

import jax.lax as lax
from typing import Any # Added import

from PDE import solve_PDE

class solution(eqx.Module):
    omega: float
    gamma: float
    delt: float
    epsilon: float
    pdeSol: Any

    def __init__(self, omega, delt=None, gamma=5/3):
        
        self.omega = omega
        self.gamma = gamma
        self.epsilon = - self.omega

        if delt is None:
            self.delt = self._find_delta_static(self.omega, self.gamma)
        else:
            self.delt = delt

        self.pdeSol = solve_PDE(self.omega, self.delt, gamma=self.gamma)
    
    @staticmethod
    def _find_delta_static(omega, gamma):
        def true_branch(args): # omega <= 3
            om, gam = args
            return (om - 3) / 2
            
        def false_branch_1(args): # omega > 3
            def true_branch_2_fn(args): # omega <= 3.2554
                return 0.0
             
            def false_branch_2_fn(args): # omega > 3.2554
                om, gam = args
                 
                def fn(delt, _args):
                    omega, gamma = _args
                    HH = (omega - 2*delt)/gamma
                    term_sqrt = jnp.maximum((delt + 2 + HH)**2 - 8 * HH, 0.0)
                    sing_U = (delt + 2 + HH  - jnp.sqrt(term_sqrt)) / 4
                    
                    num_sol = solve_PDE(omega, delt, gamma=gamma, stop_at_sonic=True)
                    last_U = num_sol.ys[-1, 0]
                    return sing_U - last_U

                solver = optx.Bisection(rtol=1e-5, atol=1e-5)
                point = optx.root_find(fn, solver, y0=0.1, args=(om, gam), throw=False, options=dict(lower=jnp.array(0.0), upper=jnp.array(14.0)))
                return jnp.array(point.value) # Ensure scalar trace

            om, gam = args
            return lax.cond(om <= 3.2554, true_branch_2_fn, false_branch_2_fn, (om, gam))

        om, gam = (omega, gamma)
        # We need to wrap first branch to accept args too
        def true_branch_wrapper(args):
            om, gam = args
            return (om - 3) / 2
            
        return lax.cond(om <= 3, true_branch_wrapper, false_branch_1, (om, gam))

    @jax.jit
    def U(self, xi):
        is_scalar = jnp.ndim(xi) == 0
        xi_arr = jnp.atleast_1d(xi)
        
        def eval_single(t):
             # evaluate returns [U, C]
             return self.pdeSol.evaluate(t)[0]
             
        res = jax.vmap(eval_single)(xi_arr)
        
        return res.reshape(jnp.shape(xi))

    @jax.jit
    def C(self, xi):
        is_scalar = jnp.ndim(xi) == 0
        xi_arr = jnp.atleast_1d(xi)
        
        def eval_single(t):
             return self.pdeSol.evaluate(t)[1]
             
        res = jax.vmap(eval_single)(xi_arr)
        return res.reshape(jnp.shape(xi))

    @jax.jit
    def G(self, xi):
        lambd = (2 * self.delt + self.omega * (self.gamma - 1)) / (3 - self.omega)
        
        const = (((1 - 2/(self.gamma + 1)) ** lambd) * (((self.gamma + 1)/(self.gamma - 1)) ** (self.gamma + 1 + lambd)) )/(2 * self.gamma * (self.gamma - 1))
        
        G = (const * (self.C(xi) ** 2) * (xi ** (2 - 3 * lambd)) * ((1 - self.U(xi)) ** (-lambd))) ** (1/(self.gamma - 1 + lambd))
        return G

    @jax.jit
    def P(self, xi):
        # Calculate P on the fly for a given xi
        C_val = self.C(xi)
        G_val = self.G(xi)
        
        P = (xi ** 2) * G_val * (C_val ** 2) / self.gamma
        return P
    
    @jax.jit
    def dUdx(self, xi):
        U = self.U(xi)
        C = self.C(xi) 

        delta0 = C ** 2 - (1 - U) ** 2
        delta1 = U * (1 - U) * (1 - U - self.delt) - (C ** 2) * (3 * U + (-self.omega + 2 * self.delt) / self.gamma)

        return delta1/(delta0 * xi)
    @jax.jit
    def dCdx(self, xi):
        U = self.U(xi)
        C = self.C(xi)

        delta0 = C ** 2 - (1 - U) ** 2
        delta2 = C * (1 - U) * (1 - U - self.delt) - (self.gamma - 1) * C * U * (2 - 2 * U + self.delt) / 2 - (C ** 3) + (2 * self.delt + (self.gamma - 1) * self.omega) * (C ** 3)/(2 * self.gamma * (1 - U))
        return delta2/(delta0 * xi)

    @jax.jit
    def dGdx(self, xi):
        lambd = (2 * self.delt + self.omega * (self.gamma - 1)) / (3 - self.omega)

        C = self.C(xi)
        U = self.U(xi)
        G = self.G(xi)

        C_term = 2 *self.dCdx(xi)/C
        U_term = lambd * self.dUdx(xi)/(1 - U)
        xi_term = (2 - 3 * lambd)/xi

        sum_terms = C_term + U_term + xi_term

        return sum_terms * G / (self.gamma - 1 + lambd)

    @jax.jit
    def dPdx(self, xi):
        G = self.G(xi)
        C = self.C(xi)
        P = self.P(xi)

        C_term = 2 *self.dCdx(xi)/C
        G_term = self.dGdx(xi)/G
        xi_term = 2/xi

        return (C_term + G_term + xi_term) * P
    


if __name__ == "__main__":
    gamma = 5/3
    omegas = [3, 3.05, 3.1, 3.15, 3.2, 3.25]

    xi_plot = jnp.linspace(1, 0, 1000)

    plt.figure()

    for omega in omegas:
        sol = solution(omega, gamma=gamma)

        U_val = sol.U(xi_plot)
        C_val = sol.C(xi_plot)
        plt.plot(U_val, C_val)

    plt.plot(xi_plot, 1 - xi_plot)
    plt.xlabel('U')
    plt.ylabel('C')
    plt.xlim(0.6, 1.0)
    plt.ylim(0.0, 0.6)
    plt.grid()

    # Create the directory if it doesn't exist
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plot_path = f'plots/solution_U_vs_C_omega_{omega}.png'
    #plt.savefig(plot_path)
    plt.show()
    print(f"Plot saved to {plot_path}")
    print("Final delta:", sol.delt)



