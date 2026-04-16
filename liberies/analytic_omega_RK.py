import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from solution_log import solution, solve_PDE 

jax.config.update("jax_enable_x64", True)

def analyze_stability(sol, l_points=200, s_points=200, eps_val=1e-4):
    """
    Runs the stability analysis for a given solution object.
    Expects `sol` to have attributes: gamma, delt, omega, xi, U, G, P, dUdxi, dGdxi, dPdxi, and sonic_xi.
    """
    gamma = sol.gamma
    delt = sol.delt
    omega = sol.omega
    sonic_point = sol.sonic_xi # Can be a scalar float

    # Convert solution arrays to JAX arrays once
    xi_arr = jnp.array(sol.xi)
    U_arr = jnp.array(sol.U)
    G_arr = jnp.array(sol.G)
    P_arr = jnp.array(sol.P)
    dU_arr = jnp.array(sol.dUdxi)
    dG_arr = jnp.array(sol.dGdxi)
    dP_arr = jnp.array(sol.dPdxi)

    # Note: jnp.interp expects the x-coordinates (xi) to be monotonically increasing.
    # If your xi goes from 1 down to sonic_xi, you must reverse the arrays for interp.
    if xi_arr[0] > xi_arr[-1]:
        xi_arr = xi_arr[::-1]
        U_arr = U_arr[::-1]
        G_arr = G_arr[::-1]
        P_arr = P_arr[::-1]
        dU_arr = dU_arr[::-1]
        dG_arr = dG_arr[::-1]
        dP_arr = dP_arr[::-1]

    @jax.jit
    def U_interp(xi): return jnp.interp(xi, xi_arr, U_arr)
    @jax.jit
    def G_interp(xi): return jnp.interp(xi, xi_arr, G_arr)
    @jax.jit
    def P_interp(xi): return jnp.interp(xi, xi_arr, P_arr)
    @jax.jit
    def dUdxi_interp(xi): return jnp.interp(xi, xi_arr, dU_arr)
    @jax.jit
    def dGdxi_interp(xi): return jnp.interp(xi, xi_arr, dG_arr)
    @jax.jit
    def dPdxi_interp(xi): return jnp.interp(xi, xi_arr, dP_arr)

    @jax.jit
    def NNr(xi):
        U = U_interp(xi)
        G = G_interp(xi)
        P = P_interp(xi)
        dUdxi = dUdxi_interp(xi)
        dGdxi = dGdxi_interp(xi)
        dPdxi = dPdxi_interp(xi)

        NN00 = omega - 3 * U - xi * dUdxi
        NN01 = - xi * dGdxi - 3 * G
        NN02 = 0.0
        NN03 = 0.0
        
        NN10 = dPdxi/G
        NN11 = (1 - delt - 2 * U - xi * dUdxi) * G * xi
        NN12 = 0.0
        NN13 = 0.0
        
        NN20 = 0.0
        NN21 = 0.0
        NN22 = (1 - delt - 2 * U) * G * xi
        NN23 = -1/xi
        
        NN30 = - gamma * (U - 1) * xi * dGdxi/(G**2)
        NN31 = gamma * xi * dGdxi/G
        NN32 = 0.0
        NN33 = xi * (U - 1) * dPdxi/(P ** 2)
        
        return jnp.array([[NN00, NN01, NN02, NN03],
                          [NN10, NN11, NN12, NN13],
                          [NN20, NN21, NN22, NN23],
                          [NN30, NN31, NN32, NN33]])

    @jax.jit
    def MM(xi):
        U = U_interp(xi)
        G = G_interp(xi)
        P = P_interp(xi)

        MM00 = xi * (U - 1)
        MM01 = G * xi
        MM02 = 0.0
        MM03 = 0.0
        
        MM10 = 0.0
        MM11 = (U - 1) * G * xi**2
        MM12 = 0.0
        MM13 = 1.0
        
        MM20 = 0.0
        MM21 = 0.0
        MM22 = (U - 1) * G * xi**2
        MM23 = 0.0
        
        MM30 = - gamma * (U - 1) * xi / G
        MM31 = 0.0
        MM32 = 0.0
        MM33 = xi * (U - 1) / P

        return jnp.array([[MM00, MM01, MM02, MM03],
                          [MM10, MM11, MM12, MM13],
                          [MM20, MM21, MM22, MM23],
                          [MM30, MM31, MM32, MM33]])

    @jax.jit
    def NNq(xi):
        G = G_interp(xi)
        P = P_interp(xi)
        return jnp.array([[-1.0, 0.0, 0.0, 0.0],
                          [0.0, - G * xi, 0.0, 0.0],
                          [0.0, 0.0, - G * xi, 0.0],
                          [gamma / G, 0.0, 0.0, - 1.0 / P]])

    @jax.jit
    def NNl(xi):
        G = G_interp(xi)
        return jnp.array([[0.0, 0.0, G, 0.0],
                          [0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0]])

    @jax.jit
    def boundary_condition(q):
        dG = dGdxi_interp(1.0)
        dU = dUdxi_interp(1.0)
        dP = dPdxi_interp(1.0)

        dG_boundary = - omega * (gamma + 1)/(gamma - 1) - dG
        dUr_boundary = 2 * q / (gamma + 1) - dU
        dUt_boundary = - 2 / (gamma + 1)
        dP_boundary = 2 * (2 * (q + 1) - omega) / (gamma + 1) - dP
        
        return jnp.array([dG_boundary, dUr_boundary, dUt_boundary, dP_boundary])

    @jax.jit
    def Y_sonic(q, l, eps=1e-5):
        from jax.experimental.ode import odeint
        last_xi = sonic_point + eps 
        Y0 = boundary_condition(q)

        def ode_system_forward(Y, t):
            xi = 1.0 - t
            NN_mat = NNr(xi) + (q * NNq(xi)) + (l * (l + 1) * NNl(xi))
            return -1.0 * jnp.linalg.solve(MM(xi), NN_mat @ Y)

        t_eval = jnp.array([0.0, 1.0 - last_xi])
        sol_ode = odeint(ode_system_forward, Y0, t_eval, rtol=1e-6, atol=1e-8)
        return sol_ode[-1]

    @jax.jit
    def Y_end(q, l, Y, eps=1e-5):
        xi = sonic_point + eps
        NN_mat = NNr(xi) + (q * NNq(xi)) + (l * (l + 1) * NNl(xi))

        P = P_interp(xi)
        G = G_interp(xi)
        U = U_interp(xi)

        vec = NN_mat @ Y
        first = - gamma * P * vec[0]
        second = xi * (U - 1) * G * vec[1]
        third = - P * vec[3]

        return first + second + third

    # Define the parameter grids
    l_arr = jnp.logspace(-3, 0, l_points)
    s_arr = jnp.linspace(-1, 0, s_points)
    alpha = 1 / (1 - delt)
    q_arr = s_arr / alpha

    @jax.jit
    def compute_single(q, l):
        Y_S = Y_sonic(q, l, eps=eps_val)
        return Y_end(q, l, Y_S, eps=eps_val)

    print(f"Starting computation for omega = {omega}...")
    vectorized_compute = jax.vmap(jax.vmap(compute_single, in_axes=(None, 0)), in_axes=(0, None))
    Y_arr = vectorized_compute(q_arr, l_arr)
    print("Computation finished.")

    return s_arr, l_arr, Y_arr

if __name__ == "__main__":
    # Example usage:
    # 1. Obtain a solution object from your solution_log library
    my_sol = solve_PDE(omega=4.25, gamma=5/3, delt=0.25)
    
    # 2. Run the analysis
    s_arr, l_arr, Y_arr = analyze_stability(my_sol)
    
    # 3. Plot the result
    Q, L = np.meshgrid(s_arr, l_arr, indexing='ij')
    Y_arr_log = np.log(np.abs(Y_arr))
    
    plt.figure()
    plt.pcolormesh(L, Q, Y_arr_log, shading='auto', cmap='viridis')
    plt.colorbar(label='Difference Norm')
    plt.xlabel('l')
    plt.xscale('log')
    plt.ylabel('q')
    plt.title(f'Y_arr for omega={my_sol.omega}')
    plt.show()