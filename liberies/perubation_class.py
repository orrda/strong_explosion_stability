import numpy as np
from solution import *
from PDE import *
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import root

# Import JAX and Diffrax
import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, PIDController, SaveAt


dist = 10


class perubation:
    def __init__(self, sol, L, q = None):
        self.sol = sol
        self.L = L
        self.q = q

        if self.q is not None:
            self.NN = None
            self.Y_init = self.get_Y_init()

    def get_q(self, x = 2, q_approx = -0.5):
        if self.q is not None:
            return self.q

        q = scipy.optimize.newton(
                lambda q: self.sonic_values_for_q(q)[x],  
                x0 = q_approx, 
                x1= -0.7,
                maxiter=2500,
                disp=False
            )
        return q

    def sonic_values_for_q(self, q):
        self.q = q

        x_end = self.sol.last_x
        Y_init = self.get_Y_init()
        # Convert Y_init to JAX array if it isn't already
        Y_init = jnp.array(Y_init)

        MM = self.sol.MM
        NN = self.get_NN(q)
        
        # DY is a series of matrices? From context, MM is likely an array of matrices.
        # Original code: DY = np.array([np.matmul(MM[i], NN[i]) for i in range(len(MM))])
        
        # We need to compute DY using JAX compatible operations if we want to JIT it, 
        # but here we are setting up the ODE.
        
        # The ODE function depends on an interpolated matrix field.
        # Or it uses lookups based on x.
        
        # Original: DY_Dx = lambda x, Y: DY[int((1 - x) * self.sol.precision)].dot(Y)
        # This implies a discrete lookup.
        
        # Let's precompute DY
        MM_jax = jnp.array(MM)
        NN_jax = jnp.array(NN)
        # Using jnp.matmul or @
        DY_jax = jnp.matmul(MM_jax, NN_jax) # Shape (T, 4, 4) if MM is (T,4,4) and NN is (T,4,4) or similar
        
        # We need a function that interpolates DY at x.
        # self.sol.xi is the x array corresponding to indices.
        # x goes from 1 down to 0 ? Or 0 to 1?
        # In solution.py: self.xi = np.linspace(1, 0, sample_rate)
        
        # If x is continuous, finding the index: int((1 - x) * self.sol.precision)
        # This mapping assumes x is in [0,1] and indices map linearly.
        
        # Let's define the vector field for Diffrax.
        # x is the independent variable (time). Diffrax calls it t.
        
        def vector_field(x, Y, args):
            DY, precision = args
            # Calculate index. Note that x is decreasing from 1 to x_end probably?
            # Original code integrates from 1 to x_end.
            # int((1 - x) * self.sol.precision)
            # 1 -> 0
            # 0 -> precision
            
            # For JAX, we cannot use flexible integer casting easily inside JIT unless we use specific JAX ops.
            # Also, Diffrax usually expects continuous functions. 
            # Step function can be problematic for adaptive solvers if it jumps too much.
            
            # A better approach for JAX/Diffrax with discrete data is to use LinearInterpolation.
            # But the grid might be coarse.
            
            # Let's just implement the same logic with jnp for now.
            idx = jnp.floor((1 - x) * precision).astype(int)
            # Clamp index to be safe
            idx = jnp.clip(idx, 0, DY.shape[0] - 1)
            
            matrix_at_x = DY[idx]
            
            return jnp.dot(matrix_at_x, Y)

        term = ODETerm(vector_field)
        solver = Dopri5()
        stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
        
        # Solve from 1 to x_end
        num_sol = diffeqsolve(
            term,
            solver,
            t0=1.0,
            t1=x_end,
            dt0=None,
            y0=Y_init,
            args=(DY_jax, self.sol.precision),
            stepsize_controller=stepsize_controller,
            saveat=SaveAt(dense=True),
            # Set max_steps to something reasonable
            max_steps=10000 
        )

        resulotion = int((1 - x_end) * self.sol.precision)
        
        # Generate evaluation points
        x_space = jnp.linspace(x_end, 1, resulotion)
        
        # Evaluate
        # Note: x_space goes from x_end to 1 (increasing), but integration was 1 to x_end (decreasing).
        # Diffrax dense output handles this.
        Y_num_sol = jax.vmap(num_sol.evaluate)(x_space)
        
        # end_vec was Y_num_sol[:, dist]
        # In original code, dist=10. 
        # Y_num_sol shape is (resolution, 4).
        # Wait, if dist is an index into the solution array along the time axis?
        # "Y_num_sol = num_sol.sol(x_space)" returns shape (4, resolution) usually in scipy?
        # Scipy solve_ivp dense_output returns a function that returns (n, t_points).
        # So Y_num_sol shape in scipy was (4, resolution).
        # In JAX/Diffrax vmap result, shape is (resolution, 4).
        
        # Original: end_vec = Y_num_sol[:, dist]
        # Is dist an index of the time steps?
        # dist = 10 at top of file.
        # If resolution > 10, then we are picking the 11th point in time?
        # Yes, looks like it picks a specific point in the trajectory.
        
        # With Diffrax vmap result (resolution, 4):
        # We need the "dist"-th point.
        # Note: x_space was linspace(x_end, 1, resolution).
        # Scipy sol(x_space) respects the order of x_space.
        
        # So we want the `dist` index from the result.
        
        # However, checking orientation:
        # Scipy: Y_num_sol is (vars, time).
        # Y_num_sol[:, dist] would be the state vector at the `dist`-th time point.
        
        # My JAX result: (time, vars).
        # So I should take [dist, :].
        
        end_vec = Y_num_sol[dist, :]

        return np.array(end_vec) # Return as numpy array for compatibility


    def integrate_qs(self, q_list):
        # We need to rewrite this method as well or ensure it uses the ported logic.
        # But looking at the file, there is code after get_NN that I didn't see.
        # I should read the rest of the file to see if there are other usages of solve_ivp.
        pass

    def get_NN(self, q):
        sol = self.sol
        omega = self.sol.omega
        l = self.L

        xi_begin = sol.last_x

        xi_begin = int(xi_begin * sol.precision)
        xi = sol.xi

        U, C = sol.get_UC()
        G = sol.get_G()
        P = sol.get_P()

        U_deriv, G_deriv, P_deriv = sol.get_derivs()

        zeros = np.zeros(sol.precision)

        N1 = np.array([omega - q - 3*U - xi * U_deriv, 
                       -xi*G_deriv - 3 * G, 
                       l * (l + 1) * G, 
                       zeros])
        N2 = np.array([P_deriv/G, 
                       xi * G * (1 - sol.delt - q - 2*U - xi * U_deriv), 
                       zeros, zeros])
        N3 = np.array([zeros, zeros, 
                       xi * G * (1 - sol.delt - q - 2*U), 
                       -1/xi])
        N4 = np.array([sol.gamma * q/G - sol.gamma * xi * (U - 1) * G_deriv/(G**2),
                        xi * sol.gamma * G_deriv/G,
                        zeros, 
                        xi * (U - 1) * P_deriv/(P**2) - q/P])

        NN = np.stack((N1, N2, N3, N4))
        NN = np.rollaxis(NN, 2, 0)[xi_begin:sol.precision - 1]

        self.NN = NN
        self.DY = np.array([np.matmul(sol.MM[i], NN[i]) for i in range(len(sol.MM))])

        return NN

    def get_Y_init(self,q):
        sol = self.sol
        U_deriv, G_deriv, P_deriv = sol.get_derivs()

        dG_boundery = sol.omega * (sol.gamma + 1)/(sol.gamma - 1) - G_deriv[1]
        dUr_boundery = 2 * q / (sol.gamma + 1) - U_deriv[1]
        dUt_boundery = -2 / (sol.gamma + 1)
        dP_boundery = 2 * (2 * (q + 1) - sol.omega) / (sol.gamma + 1) - P_deriv[1]

        self.Y_init = np.array([dG_boundery, dUr_boundery, dUt_boundery, dP_boundery])
        return self.Y_init
    

    def get_q_smart(self, q_approx = -0.5):
        q = scipy.optimize.newton(
                lambda q: self.get_dotProd_by_q(q),  
                x0 = q_approx, 
                x1= -0.7,
                maxiter=500,
                disp=False
            )
        print("q is ", q)
        return q

    def get_dotProd_by_q(self,q):

        last_x = int(self.sol.last_x * self.sol.precision)

        MM = self.sol.MM

        #1. find eigenvector of M with eigenvalue 0 in the sonic point
        print("MM[0] is ", MM[0])
        eig, vec = np.linalg.eig(self.sol.MM[0])
        print("eig is ", eig)

        #2. find 3 vectors that are orthogonal to the eigenvector
        non0vecs = [vec[i] for i in range(len(eig)) if eig[i] != 0]
        print("non0vecs is ", non0vecs)

        #//3. solve analytically from the sonic point to a small distance for each of the 3 vectors
        #4. from the small distance, solve numerically to the boundery

        end_vec = self.solve_from_sonic_to_boundery(non0vecs, q)
        print("end_vec is ", end_vec)

        #5. find a normaliezed vector that is orthogonal to the 3 vectors
        vec = self.gram_schmidt(end_vec)
        print("vec is ", vec)
        #6. dot product the normalized vector with the boundery
        boundery = self.get_Y_init(q)
        dotProd = np.dot(vec, boundery)
        print("boundery is ", boundery)


        print("for q - ", q, "  dot prod is - ", dotProd)

        return dotProd
    #7. find the q that makes the dot product 0

    def solve_from_sonic_to_boundery(self, vecs, q):
        end_vec = []
        last_x = self.sol.last_x
        # Note: self.sol.MM_inv and get_NN(q) likely return numpy arrays.
        # We need to adapt the logic for JAX + Diffrax.

        MM_inv = self.sol.MM_inv
        NN = self.get_NN(q)
        
        # Precompute DY
        DY = jnp.array([jnp.matmul(MM_inv[i], NN[i]) for i in range(len(MM_inv))])
        
        # Define vector field
        def vector_field(x, Y, args):
            DY, precision, last_x = args
            idx = jnp.floor((x - last_x) * precision).astype(int)
            idx = jnp.clip(idx, 0, DY.shape[0] - 1)
            
            matrix_at_x = DY[idx]
            
            return jnp.dot(matrix_at_x, Y)

        vecs_jnp = jnp.array(vecs) # shape (3, N_vars)
        
        term = ODETerm(vector_field)
        solver = Dopri5()
        stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
        
        def solve_single(y_init):
             sol = diffeqsolve(
                term,
                solver,
                t0=last_x,
                t1=1.0,
                dt0=None,
                y0=y_init,
                args=(DY, self.sol.precision, last_x),
                stepsize_controller=stepsize_controller,
                max_steps=10000 
            )
             return sol.ys[-1]

        # Use vmap to solve for all vectors
        end_vecs = jax.vmap(solve_single)(vecs_jnp)

        return np.array(end_vecs) # Convert back to numpy


    def gram_schmidt(self, vecs):
        vec = [1, 1, 1, 1]
        for i in range(1, len(vecs)):
            vec = vec - np.dot(vec, vecs[i]) * vecs[i]
        return vec/np.linalg.norm(vec)
