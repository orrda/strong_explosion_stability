import numpy as np
import scipy.integrate
from solution2 import *
from PDE import *
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import root


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
        
        func = lambda q: abs(self.get_sonic_val(q)[3])


        q1 = scipy.optimize.minimize_scalar(
                func,
                bounds = (-3, 3)
            )


        return q1.x



    def q_space(self):
        Re_lower = 0.025
        Re_upper = 0.075
        Im_lower = -1
        Im_upper = 1



        q_Re_space = np.linspace(Re_lower, Re_upper, 200)
        q_Im_space = np.linspace(Im_lower, Im_upper, 8)
        Y_space = np.zeros((len(q_Re_space), len(q_Im_space), 4))
        for i, q_Re in enumerate(q_Re_space):
            for j, q_Im in enumerate(q_Im_space):
                q = q_Re + 1j * q_Im
                Y = self.get_sonic_val(q)
                Y_space[i, j] = Y
        
        plt.imshow(np.abs(Y_space[:, :, 3]), extent=(Im_lower, Im_upper, Re_lower, Re_upper), aspect='auto', origin='lower', interpolation='nearest')
        plt.colorbar(label='|Y[3]|')
        plt.ylabel('Re(q)')
        plt.xlabel('Im(q)')
        plt.title('Sonic Value |Y[3]| as a function of q')
        plt.show()
        return Y_space

                

    def get_sonic_val(self, q):
        Y_init = self.get_Y_init(q)
        mat = self.get_DYDxi(q)
        Y_0 = mat.dot(Y_init)
        print(f"Y_0: {Y_0}")
        return Y_0



    def get_DYDxi(self, q):
        sol = self.sol

        if sol.last_x is None:
            sol.last_x = sol.last_X()
        xi_begin = int(sol.precision * (1 - sol.last_x))

        U, C = sol.get_UC()
        xi = sol.xi[:xi_begin]
        U = U[:xi_begin]
        G = sol.get_G()[:xi_begin]
        P = sol.get_P()[:xi_begin]
        U_deriv, G_deriv, P_deriv = sol.get_derivs()
        U_deriv = U_deriv[:xi_begin]
        G_deriv = G_deriv[:xi_begin]
        P_deriv = P_deriv[:xi_begin]
        delta = sol.delt
        gamma = sol.gamma
        omega = sol.omega
        l = self.L



        # Avoid division by zero
        eps = 1e-20
        xi = np.array(xi)
        U = np.array(U)
        G = np.array(G)
        P = np.array(P)
        U_deriv = np.array(U_deriv)
        G_deriv = np.array(G_deriv)
        P_deriv = np.array(P_deriv)

        A = np.zeros((len(xi), 4, 4))

        num1 = -(gamma*xi*P*(U-1)*G_deriv) - xi**2*G**2*(U-1)**2*(q-omega+3*U+xi*U_deriv) + G*(q*gamma*P - xi*(U-1)*P_deriv)
        den1 = xi*G*(-(gamma*P) + xi**2*G*(U-1)**2)*(U-1)
        A[:,0,0] = num1 / (den1 + eps)

        num2 = gamma*P*G_deriv - xi**2*G*(U-1)**2*G_deriv + xi*G**2*(U-1)*(2+q+delta-U+xi*U_deriv)
        den2 = (-(gamma*P) + xi**2*G*(U-1)**2)*(U-1)
        A[:,0,1] = num2 / (den2 + eps)

        A[:,0,2] = (l*(1+l)*xi*G**2*(U-1)) / (-(gamma*P) + xi**2*G*(U-1)**2 + eps)

        A[:,0,3] = (G*(-(q*P) + xi*(U-1)*P_deriv)) / (xi*P*(-(gamma*P) + xi**2*G*(U-1)**2)*(U-1) + eps)

        num4 = gamma*P*(xi*(U-1)*G_deriv + G*(-omega+3*U+xi*U_deriv)) + xi*G*(U-1)*P_deriv
        den4 = xi*G**2*(-(gamma*P) + xi**2*G*(U-1)**2)
        A[:,1,0] = num4 / (den4 + eps)

        A[:,1,1] = (3*gamma*P - xi**2*G*(U-1)*(-1+q+delta+2*U+xi*U_deriv)) / (-(gamma*xi*P) + xi**3*G*(U-1)**2 + eps)

        A[:,1,2] = -((l*(1+l)*gamma*P) / (-(gamma*xi*P) + xi**3*G*(U-1)**2 + eps))

        A[:,1,3] = (-(q*P) + xi*(U-1)*P_deriv) / (xi*P*(gamma*P - xi**2*G*(U-1)**2) + eps)

        A[:,2,0] = 0
        A[:,2,1] = 0
        A[:,2,2] = -((-1+q+delta+2*U) / (xi*(U-1) + eps))
        A[:,2,3] = -(1/(xi**3*G*(U-1) + eps))

        num7 = -gamma*P*(xi*(U-1)*(xi*(U-1)*G_deriv + G*(-omega+3*U+xi*U_deriv)) + P_deriv)
        den7 = G*(-(gamma*P) + xi**2*G*(U-1)**2)
        A[:,3,0] = num7 / (den7 + eps)

        A[:,3,1] = -((gamma*xi*G*P*(2+q+delta-U+xi*U_deriv)) / (gamma*P - xi**2*G*(U-1)**2 + eps))

        A[:,3,2] = -((l*(1+l)*gamma*xi*G*P*(U-1)) / (gamma*P - xi**2*G*(U-1)**2 + eps))

        A[:,3,3] = (xi*G*(U-1)*(q*P - xi*(U-1)*P_deriv)) / (P*(gamma*P - xi**2*G*(U-1)**2) + eps)

        A = A/len(xi)

        eyes = np.eye(4)
        eyes = np.repeat(eyes[np.newaxis, :, :], len(xi), axis=0)
        A = A + eyes

        A_product = np.linalg.multi_dot(A)

        return A_product



    def get_Y_init(self,q):
        sol = self.sol
        U_deriv, G_deriv, P_deriv = sol.get_derivs()

        dG_boundery = sol.omega * (sol.gamma + 1)/(sol.gamma - 1) - G_deriv[1]
        dUr_boundery = 2 * q / (sol.gamma + 1) - U_deriv[1]
        dUt_boundery = -2 / (sol.gamma + 1)
        dP_boundery = 2 * (2 * (q + 1) - sol.omega) / (sol.gamma + 1) - P_deriv[1]

        self.Y_init = np.array([dG_boundery, dUr_boundery, dUt_boundery, dP_boundery])
        return self.Y_init
    

    def get_q_new(self, q_guess):
        if self.sol.last_x is None:
            self.sol.last_x = self.sol.last_X()
        last_x = int(self.sol.last_x * self.sol.precision)

        normlizer = 100000

        dist = 1

        while dist < last_x:

            interval = 1
            DMDq = self.get_DMDq(q_guess, dist)
            if DMDq == np.nan or np.isinf(DMDq):
                print(f"NaN encountered for q_guess: {q_guess} at dist: {dist}")
                return q_guess
            q_guess = q_guess + interval * DMDq / normlizer
            dist = int(dist + interval)

        print(f"Final q_guess: {q_guess}")
        self.q = q_guess
        return q_guess
        



    def get_DMDq(self, q, dist):
        MM = self.get_M(q, dist)
        QQ = self.get_Q(q, dist)

        Y_init = self.get_Y_init(q)
        P_vec = np.array([0, 0, 0, 1])

        MM_beg = []
        for i in range(len(MM)):
            Y_init = MM[i].dot(Y_init)
            MM_beg.append(Y_init)
        MM_beg = np.array(MM_beg)

        MM_end = []
        for i in range(len(MM)):
            P_vec = P_vec.dot(MM[i])
            MM_end.append(P_vec)
        MM_end = np.array(MM_end)

        scalar_arr = []
        len_QQ = len(QQ)
        for i in range(len_QQ):
            scalar = MM_end[len_QQ - i - 1].dot(QQ[i]).dot(MM_beg[i-1])
            scalar_arr.append(scalar)

        scalar_arr = np.array(scalar_arr)
        final_scalar = np.sum(scalar_arr)

        return final_scalar/dist
    

    def get_M(self, q, dist):

        sol = self.sol

        U, C = sol.get_UC()
        xi = sol.xi[:dist]
        U = U[:dist]
        G = sol.get_G()[:dist]
        P = sol.get_P()[:dist]
        U_deriv, G_deriv, P_deriv = sol.get_derivs()
        U_deriv = U_deriv[:dist]
        G_deriv = G_deriv[:dist]
        P_deriv = P_deriv[:dist]
        delta = sol.delt
        gamma = sol.gamma
        omega = sol.omega
        l = self.L



        # Avoid division by zero
        eps = 1e-20
        xi = np.array(xi)
        U = np.array(U)
        G = np.array(G)
        P = np.array(P)
        U_deriv = np.array(U_deriv)
        G_deriv = np.array(G_deriv)
        P_deriv = np.array(P_deriv)

        A = np.zeros((len(xi), 4, 4))

        num1 = -(gamma*xi*P*(U-1)*G_deriv) - xi**2*G**2*(U-1)**2*(q-omega+3*U+xi*U_deriv) + G*(q*gamma*P - xi*(U-1)*P_deriv)
        den1 = xi*G*(-(gamma*P) + xi**2*G*(U-1)**2)*(U-1)
        A[:,0,0] = num1 / (den1 + eps)

        num2 = gamma*P*G_deriv - xi**2*G*(U-1)**2*G_deriv + xi*G**2*(U-1)*(2+q+delta-U+xi*U_deriv)
        den2 = (-(gamma*P) + xi**2*G*(U-1)**2)*(U-1)
        A[:,0,1] = num2 / (den2 + eps)

        A[:,0,2] = (l*(1+l)*xi*G**2*(U-1)) / (-(gamma*P) + xi**2*G*(U-1)**2 + eps)

        A[:,0,3] = (G*(-(q*P) + xi*(U-1)*P_deriv)) / (xi*P*(-(gamma*P) + xi**2*G*(U-1)**2)*(U-1) + eps)

        num4 = gamma*P*(xi*(U-1)*G_deriv + G*(-omega+3*U+xi*U_deriv)) + xi*G*(U-1)*P_deriv
        den4 = xi*G**2*(-(gamma*P) + xi**2*G*(U-1)**2)
        A[:,1,0] = num4 / (den4 + eps)

        A[:,1,1] = (3*gamma*P - xi**2*G*(U-1)*(-1+q+delta+2*U+xi*U_deriv)) / (-(gamma*xi*P) + xi**3*G*(U-1)**2 + eps)

        A[:,1,2] = -((l*(1+l)*gamma*P) / (-(gamma*xi*P) + xi**3*G*(U-1)**2 + eps))

        A[:,1,3] = (-(q*P) + xi*(U-1)*P_deriv) / (xi*P*(gamma*P - xi**2*G*(U-1)**2) + eps)

        A[:,2,0] = 0
        A[:,2,1] = 0
        A[:,2,2] = -((-1+q+delta+2*U) / (xi*(U-1) + eps))
        A[:,2,3] = -(1/(xi**3*G*(U-1) + eps))

        num7 = -gamma*P*(xi*(U-1)*(xi*(U-1)*G_deriv + G*(-omega+3*U+xi*U_deriv)) + P_deriv)
        den7 = G*(-(gamma*P) + xi**2*G*(U-1)**2)
        A[:,3,0] = num7 / (den7 + eps)

        A[:,3,1] = -((gamma*xi*G*P*(2+q+delta-U+xi*U_deriv)) / (gamma*P - xi**2*G*(U-1)**2 + eps))

        A[:,3,2] = -((l*(1+l)*gamma*xi*G*P*(U-1)) / (gamma*P - xi**2*G*(U-1)**2 + eps))

        A[:,3,3] = (xi*G*(U-1)*(q*P - xi*(U-1)*P_deriv)) / (P*(gamma*P - xi**2*G*(U-1)**2) + eps)
        

        xi_max = np.max(xi)
        A = A * xi_max / len(xi)

        eyes = np.eye(4)
        eyes = np.repeat(eyes[np.newaxis, :, :], len(xi), axis=0)
        A = A + eyes

        return A


    

    def get_Q(self, q, dist):
        sol = self.sol

        U, C = sol.get_UC()
        xi = sol.xi[:dist]
        U = U[:dist]
        G = sol.get_G()[:dist]
        P = sol.get_P()[:dist]
        gamma = sol.gamma

        # Avoid division by zero
        eps = 1e-20
        xi = np.array(xi)
        U = np.array(U)
        G = np.array(G)
        P = np.array(P)

        Q = np.zeros((len(xi), 4, 4), dtype=np.complex128)

        # First row
        Q[:, 0, 0] = 1.0 / (xi - xi * U + eps)
        Q[:, 0, 1] = (xi * G**2) / (-(gamma * P) + xi**2 * G * (U - 1)**2 + eps)
        Q[:, 0, 2] = 0
        Q[:, 0, 3] = -G / (xi * (-(gamma * P) + xi**2 * G * (U - 1)**2) * (U - 1) + eps)

        # Second row
        Q[:, 1, 0] = 0
        Q[:, 1, 1] = -((xi * G * (U - 1)) / (-(gamma * P) + xi**2 * G * (U - 1)**2 + eps))
        Q[:, 1, 2] = 0
        Q[:, 1, 3] = 1.0 / (-(gamma * xi * P) + xi**3 * G * (U - 1)**2 + eps)

        # Third row
        Q[:, 2, 0] = 0
        Q[:, 2, 1] = 0
        Q[:, 2, 2] = 1.0 / (xi - xi * U + eps)
        Q[:, 2, 3] = 0

        # Fourth row
        Q[:, 3, 0] = 0
        Q[:, 3, 1] = -((gamma * xi * G * P) / (gamma * P - xi**2 * G * (U - 1)**2 + eps))
        Q[:, 3, 2] = 0
        Q[:, 3, 3] = -((xi * G * (U - 1)) / (-(gamma * P) + xi**2 * G * (U - 1)**2 + eps))
        
        xi_max = np.max(xi)
        Q = Q * xi_max / len(xi)

        return Q
    


    def DMdq_space(self, q, dist):
        MM = self.get_M(q, dist)
        QQ = self.get_Q(q, dist)

        Y_init = self.get_Y_init(q)
        P_vec = np.array([1, 0, 0, 0])


        DMdqs = [QQ[0].dot(Y_init)]
        MM_next = MM[0].dot(Y_init)
        for i in range(1,len(MM)):
            DMdq_i = MM[i].dot(DMdqs[-1]) + QQ[i].dot(MM_next)
            DMdqs.append(DMdq_i)
            MM_next = MM[i].dot(MM_next)
        DMdqs = np.array(DMdqs)
        scalar_arr = P_vec.dot(DMdqs.T)

        return scalar_arr.T
    
    def pressure_space(self, q, dist):
        MM = self.get_M(q, dist)
        Y_init = self.get_Y_init(q)
        P_vec = np.array([0, 0, 0, 1])
    
        current_MM = Y_init
        P_space = []
        for i in range(len(MM)):
            current_MM = MM[i].dot(current_MM)
            P_space.append(P_vec.dot(current_MM))
        return np.array(P_space)
