import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp  # Added JAX import
import optimistix as optx
import equinox as eqx

import jax.lax as lax
from typing import Any # Added import

from PDE_log import solve_PDE

class solution(eqx.Module):
    omega: float
    gamma: float
    delt: float
    epsilon: float
    pdeSol: Any
    xi_final: float

    def __init__(self, omega, delt=None, gamma=5/3):
        
        self.omega = omega
        self.gamma = gamma
        self.epsilon = - self.omega

        if delt is None:
            self.delt = self._find_delta_static(self.omega, self.gamma)
        else:
            self.delt = delt

        self.pdeSol = solve_PDE(self.omega, self.delt, gamma=self.gamma, stop_at_sonic=True)
        self.xi_final = self.pdeSol.ts[-1]

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

                solver = optx.Bisection(rtol=1e-12, atol=1e-12)
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
    def xi(self, t):
        t_arr = jnp.atleast_1d(t)

        res = jax.vmap(lambda t: self.pdeSol.evaluate(t)[2])(t_arr)

        return res.reshape(jnp.shape(t))

    @jax.jit
    def U(self, t):
        t_arr = jnp.atleast_1d(t)

        res = jax.vmap(lambda t: self.pdeSol.evaluate(t)[0])(t_arr)

        return res.reshape(jnp.shape(t))

    @jax.jit
    def C(self, t):
        t_arr = jnp.atleast_1d(t)

        res = jax.vmap(lambda t: self.pdeSol.evaluate(t)[1])(t_arr)

        return res.reshape(jnp.shape(t))

    @jax.jit
    def G(self, t):
        lambd = (2 * self.delt + self.omega * (self.gamma - 1)) / (3 - self.omega)
        
        const = (((1 - 2/(self.gamma + 1)) ** lambd) * (((self.gamma + 1)/(self.gamma - 1)) ** (self.gamma + 1 + lambd)) )/(2 * self.gamma * (self.gamma - 1))

        G = (const * (self.C(t) ** 2) * (self.xi(t) ** (2 - 3 * lambd)) * ((1 - self.U(t)) ** (-lambd))) ** (1/(self.gamma - 1 + lambd))

        return G

    @jax.jit
    def P(self, t):
        # Calculate P on the fly for a given xi
        C_val = self.C(t)
        G_val = self.G(t)
        
        P = (self.xi(t) ** 2) * G_val * (C_val ** 2) / self.gamma
        return P
    
    @jax.jit
    def dUdxi(self, t):
        U = self.U(t)
        C = self.C(t) 

        delta0 = C ** 2 - (1 - U) ** 2
        delta1 = U * (1 - U) * (1 - U - self.delt) - (C ** 2) * (3 * U + (-self.omega + 2 * self.delt) / self.gamma)

        return delta1/(delta0 * self.xi(t))
    @jax.jit
    def dCdxi(self, t):
        U = self.U(t)
        C = self.C(t)

        delta0 = C ** 2 - (1 - U) ** 2
        delta2 = C * (1 - U) * (1 - U - self.delt) - (self.gamma - 1) * C * U * (2 - 2 * U + self.delt) / 2 - (C ** 3) + (2 * self.delt + (self.gamma - 1) * self.omega) * (C ** 3)/(2 * self.gamma * (1 - U))
        return delta2/(delta0 * self.xi(t))

    @jax.jit
    def dGdxi(self, t):
        lambd = (2 * self.delt + self.omega * (self.gamma - 1)) / (3 - self.omega)

        C = self.C(t)
        U = self.U(t)
        G = self.G(t)

        C_term = 2 *self.dCdxi(t)/C
        U_term = lambd * self.dUdxi(t)/(1 - U)
        xi_term = (2 - 3 * lambd)/self.xi(t)

        sum_terms = C_term + U_term + xi_term

        return sum_terms * G / (self.gamma - 1 + lambd)

    @jax.jit
    def dPdxi(self, t):
        G = self.G(t)
        C = self.C(t)
        P = self.P(t)

        C_term = 2 *self.dCdxi(t)/C
        G_term = self.dGdxi(t)/G
        xi_term = 2/self.xi(t)

        return (C_term + G_term + xi_term) * P
    
    @jax.jit
    def NNr(self, t):
        U = self.U(t)
        G = self.G(t)
        P = self.P(t)
        xi = self.xi(t)
        
        dUdxi = self.dUdxi(t)
        dGdxi = self.dGdxi(t)
        dPdxi = self.dPdxi(t)

        NN00 = self.omega - 3 * U - xi * dUdxi
        NN01 = - xi * dGdxi - 3 * G
        NN02 = 0
        NN03 = 0

        NN10 = dPdxi/G
        NN11 = (1 - self.delt - 2 * U - xi * dUdxi) * G * xi
        NN12 = 0
        NN13 = 0

        NN20 = 0
        NN21 = 0
        NN22 = (1 - self.delt - 2 * U) * G * xi
        NN23 = 1/xi

        NN30 = self.gamma * (U - 1) * xi * dGdxi/G
        NN31 = self.gamma * xi * dGdxi/G
        NN32 = 0
        NN33 = xi * (U - 1) * dPdxi/(P ** 2)

        return jnp.array([[NN00, NN01, NN02, NN03],
                           [NN10, NN11, NN12, NN13],
                           [NN20, NN21, NN22, NN23],
                           [NN30, NN31, NN32, NN33]])
    

    @jax.jit
    def MM(self, t):
        U = self.U(t)
        G = self.G(t)
        P = self.P(t)
        xi = self.xi(t)

        MM00 = xi * (U - 1)
        MM01 = G * xi
        MM02 = 0
        MM03 = 0

        MM10 = 0
        MM11 = (U - 1) * G * xi**2
        MM12 = 0
        MM13 = 1

        MM20 = 0
        MM21 = 0
        MM22 = (U - 1) * G * xi**2
        MM23 = 0

        MM30 = - self.gamma * (U - 1) * xi / G
        MM31 = 0
        MM32 = 0
        MM33 = xi * (U - 1) / P

        return jnp.array([[MM00, MM01, MM02, MM03],
                           [MM10, MM11, MM12, MM13],
                           [MM20, MM21, MM22, MM23],
                           [MM30, MM31, MM32, MM33]])
    
    @jax.jit
    def NNq(self,t):
        G = self.G(t)
        P = self.P(t)
        xi = self.xi(t)

        return jnp.array([[-1, 0, 0, 0],
                           [0, - G * xi, 0, 0],
                           [0, 0, - G * xi, 0],
                           [self.gamma / G, 0, 0, - 1 / P]])
    
    def NNl(self, t):
        G = self.G(t)

        return jnp.array([[0, 0, G, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])

if __name__ == "__main__":
    gamma = 5/3
    omegas = [0.5, 1.0, 3.25, 3.2554, 4.01, 4.25, 5]

    plt.figure()

    for omega in omegas:
        sol = solution(omega, gamma=gamma)
        print(f"Omega: {omega}, Delta: {sol.delt}")

        print(sol.xi_final)

        t_arr = jnp.linspace(0, 1, 1000) * sol.xi_final

        U_val = sol.U(t_arr)
        C_val = sol.C(t_arr)

        last_U_t = sol.U(sol.xi_final)
        last_C_t = sol.C(sol.xi_final)

        plt.plot(U_val, C_val, '.')
        plt.plot(last_U_t, last_C_t, 'ro')

    line = jnp.linspace(0, 1, 100)
    plt.plot(line, 1-line, 'r--')
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



