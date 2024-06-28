import numpy as np
import scipy.integrate
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
        self.DY = None

        if self.q is not None:
            self.NN = None
            self.Y_init = self.get_Y_init(self.q)

    def get_q(self):
        if self.q is not None:
            return self.q
        
        func = lambda q: abs(self.get_sonic_val2(q)[2])
        """
        q_array = np.linspace(-1, 0, 100)
        sonic_values = np.array([func(q) for q in q_array])
        plt.plot(q_array, sonic_values[:,0], '.')
        plt.plot(q_array, sonic_values[:,1], '.')
        plt.plot(q_array, sonic_values[:,2], '.')
        plt.plot(q_array, sonic_values[:,3], '.')
        plt.show()

        return q_array[np.argmin(np.abs(sonic_values[:,0]))]
        """



        q = scipy.optimize.minimize_scalar(
                func,
                bracket = (-1, 0)
            )
        print("q is ", q.x, " for L = ", self.L, " and sonic values are - ", self.sonic_values_for_q(q.x))
        self.q = q.x
        return q.x

    
    def sonic_values_for_q(self, q):
        self.q = q

        x_end = self.sol.last_X()
        interval = 1 - x_end

        MM = self.sol.get_MM()
        NN = self.get_NN(q)

        resulotion = len(NN)
        step = interval / resulotion

        Y_init = self.get_Y_init(self.q)
        next_Y = Y_init
        next_Ys = [next_Y]


        for i in range(resulotion - 1):
            DY_Dx = np.linalg.solve(MM[i], NN[i].dot(next_Y))
            if np.linalg.norm(DY_Dx) > 10 ** 200:
                print("DY_Dx is ", DY_Dx, " and Y is ", next_Y, " and i is ", i)
                return next_Y
            next_Y = next_Y - DY_Dx*step
            next_Ys.append(next_Y)



        """
        next_Ys = np.array(next_Ys[:-1])
        plt.title("L = " + str(self.L) + " q = " + str(q))
        plt.semilogy(np.linspace(1, x_end, resulotion - 1), next_Ys[:,0], '.-', color = 'red')  
        plt.semilogy(np.linspace(1, x_end, resulotion - 1), next_Ys[:,1], '.-', color = 'blue')  
        plt.semilogy(np.linspace(1, x_end, resulotion - 1), next_Ys[:,2], '.-', color = 'green')  
        plt.semilogy(np.linspace(1, x_end, resulotion - 1), next_Ys[:,3], '.-', color = 'black')
        plt.semilogy(np.linspace(1, x_end, resulotion - 1), - next_Ys[:,0], '.-', color = 'red')  
        plt.semilogy(np.linspace(1, x_end, resulotion - 1), - next_Ys[:,1], '.-', color = 'blue')  
        plt.semilogy(np.linspace(1, x_end, resulotion - 1), - next_Ys[:,2], '.-', color = 'green')  
        plt.semilogy(np.linspace(1, x_end, resulotion - 1), - next_Ys[:,3], '.-', color = 'black')
          
        plt.legend(["dG", "dUr", "dUt", "dP"])
        plt.show()
        """
        
        return next_Y
    
    def get_sonic_val2(self, q):
        self.q = q

        x_end = self.sol.last_X()
        interval = 1 - x_end

        MM = self.sol.get_MM()
        NN = self.get_NN(q)


        resulotion = len(NN)
        step = interval / resulotion

        eye_mat = np.outer(np.ones(resulotion),np.eye(4))
        eye_mat = np.reshape(eye_mat, (resulotion, 4, 4))
        MM_inv = np.linalg.inv(MM)
        final_arr = eye_mat - step * np.matmul(MM_inv, NN)
        final_matrix = np.prod(final_arr, axis = 0)
        print("final_matrix1 is ", final_matrix)


        final_matrix = np.eye(4)
        for i in range(resulotion):
            MM_inv = np.linalg.inv(MM[i])
            final_matrix = np.matmul(final_matrix, np.eye(4) - step * np.matmul(MM_inv, NN[i]))

        print("final_matrix2 is ", final_matrix)
        Y_init = self.get_Y_init(self.q)

        return final_matrix.dot(Y_init)

    def get_NN(self, q):
        sol = self.sol
        omega = self.sol.omega
        l = self.L

        if sol.last_x is None:
            sol.last_x = sol.last_X()
        xi_begin = sol.last_x

        xi_begin = int(xi_begin * sol.precision)
        xi = sol.xi

        U, C = sol.get_UC()
        G = sol.get_G()
        P = sol.get_P()

        U_deriv = np.gradient(U, xi)
        G_deriv = np.gradient(G, xi)
        P_deriv = np.gradient(P, xi)

        zeros = np.zeros(sol.precision)
        ones = np.ones(sol.precision)

        N11 = (ones *sol.gamma * q)/G - sol.gamma * xi * (U - 1) * G_deriv/(G ** 2)


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
        N4 = np.array([ N11,
                        xi * sol.gamma * G_deriv/G,
                        zeros, 
                        xi * (U - 1) * P_deriv/(P**2) - q/P])

        NN = np.stack((N1, N2, N3, N4))
        NN = np.rollaxis(NN, 2, 0)[:xi_begin]

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
    