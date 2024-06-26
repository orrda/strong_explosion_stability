import numpy as np
from solution import *
from PDE import *
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import root


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

        MM = self.sol.MM
        NN = self.get_NN(q)
        DY = np.array([np.matmul(MM[i], NN[i]) for i in range(len(MM))])

        DY_Dx = lambda x, Y: DY[int((1 - x) * self.sol.precision)].dot(Y)

        num_sol = solve_ivp(
            DY_Dx,
            [1, x_end],
            Y_init,
            method='RK45',
            dense_output=True,
        )

        resulotion = int((1 - x_end) * self.sol.precision)

        x_space = np.linspace(x_end, 1, resulotion)
        Y_num_sol = num_sol.sol(x_space)
        end_vec = Y_num_sol[:, dist]

        return end_vec


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
        resulotion = int((1 - last_x) * self.sol.precision)
        

        MM_inv = self.sol.MM_inv
        NN = self.get_NN(q)
        DY = np.array([np.matmul(MM_inv[i], NN[i]) for i in range(len(MM_inv))])

        DY_Dx = lambda x, Y: DY[int((x - last_x) * self.sol.precision)].dot(Y)

        for vec in vecs:
            num_sol = solve_ivp(
                DY_Dx,
                [last_x, 1],
                vec,
                method='RK45',
                dense_output=True,
            )

            
            x_space = np.linspace(last_x, 1, resulotion)
            Y_num_sol = num_sol.sol(x_space)
            end_vec.append(Y_num_sol[:, -1])

        return np.array(end_vec)



    def gram_schmidt(self, vecs):
        vec = [1, 1, 1, 1]
        for i in range(1, len(vecs)):
            vec = vec - np.dot(vec, vecs[i]) * vecs[i]
        return vec/np.linalg.norm(vec)
    