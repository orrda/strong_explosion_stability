import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import scipy.optimize

from PDE import *

sample_rate = 10000

class solution:

    def __init__(self, omega, delt=None, gamma=5/3):
        """
        Initializes a new instance of the solution class.

        Parameters:
        - omega: The value of omega.
        - delt: The value of delt.
        - gamma: The value of gamma.
        """
        self.omega = omega
        self.gamma = gamma
        self.xi = np.linspace(1, 0, sample_rate)
        self.precision = sample_rate
        self.last_x = None
        self.U = None
        self.C = None
        self.G = None
        self.P = None
        if delt is None:
            self.delt = self.find_delta()
        else:
            self.delt = delt
        self.U_deriv = None
        self.C_deriv = None
        self.G_deriv = None
        self.P_deriv = None

        self.MM = None
        self.MM_inv = None
        self.W0 = None
        self.Wq = None
        self.Wl = None


    def find_delta(self):
        if self.omega <= 3:
            self.delt = (self.omega - 3) / 2
        
        if self.omega > 3 and self.omega <= 3.26:
            self.delt = 0
        
        if self.omega > 3.26:
            delta_infimum, delta_suprimum = self.check_DB(self.omega)
            print("delta_infimum is ", delta_infimum, "delta_suprimum is ", delta_suprimum)
            func = lambda delt: self.sonic_U(delt) - self.singuler_U(delt)
            if func(delta_infimum) == func(delta_suprimum):
                self.delt = delta_infimum
            else:
                if func(delta_infimum) * func(delta_suprimum) > 0:
                    while func(delta_suprimum) < 0:
                        delta_suprimum = 2*delta_suprimum - delta_infimum
                    while func(delta_infimum) > 0:
                        delta_infimum = 2*delta_infimum - delta_suprimum

                root = scipy.optimize.root_scalar(
                        func, 
                        method = 'brentq',
                        bracket = [delta_infimum,delta_suprimum],
                        maxiter = 1000,
                        xtol = 1e-10
                    )
                self.delt = root.root

                self.save_DB(self.omega, self.delt)
        return self.delt


    def check_DB(self,omega):
        path = "DB\\omega_delta.npy"
        data = np.load(path)

        omegas = data[:,0]
        deltas = data[:,1]

        smaller_omega = omegas[omegas <= omega]
        bigger_omega = omegas[omegas >= omega]

        if len(smaller_omega) == 0:
            delta_infimum = 0
        else:
            delta_infimum = max(deltas[omegas == max(smaller_omega)])

        if len(bigger_omega) == 0:
            delta_supremum = delta_infimum + 1
        else:  
            delta_supremum = min(deltas[omegas == min(bigger_omega)])

        return delta_infimum, delta_supremum

    def save_DB(self, omega, delta):
        path = "DB\\omega_delta.npy"
        data = np.load(path)
        data = np.append(data, [[omega, delta]], axis=0)
        np.sort(data, axis=0)
        np.save(path, data)



    def get_UC(self, x_space=np.linspace(1, 0, sample_rate)):
        """
        Gets the U and C arrays.

        Parameters:
        - x_space: The x space array.

        Returns:
        - The U and C arrays.
        """
        if self.U is not None and self.C is not None:
            return self.U, self.C
        num_sol = solve_PDE(self.omega, self.delt, gamma=self.gamma)
        UC_num_sol = num_sol.sol(x_space)
        self.U = UC_num_sol[0].T
        self.C = UC_num_sol[1].T
        return self.U, self.C

    def last_X(self):
        if self.omega <= 2:
            return 1
        
        if self.delt is None:
            self.find_delta()
        x_space = np.linspace(0, 1, self.precision)
        num_sol = solve_PDE(self.omega, self.delt, indi=False, x_end = 0).sol(x_space)
        U = num_sol[0].T
        C = num_sol[1].T
        delta = C**2 - (1 - U)**2
        negetive = np.where(delta <= 0)
        if len(negetive[0]) == 0:
            min_index = np.argmin(delta)
            print("delta is positive, min index is ", min_index)
        else:
            min_index = negetive[0][-1]
        return x_space[min_index]


    def save_last_x_DB(self, last_x):
        path = "DB\\last_x.npy"
        data = np.load(path)
        last = [self.omega, self.delt,last_x]
        data = np.append(data, [last], axis=0)
        np.save(path, data)

    def check_last_x_DB(self):
        path = "DB\\last_x.npy"
        data = np.load(path)
        omegas = data[:,0]
        deltas = data[:,1]
        last_xs = data[:,2]

        min = 100

        for i in range(len(omegas)):
            dist = np.sqrt((self.omega - omegas[i])**2 + (self.delt - deltas[i])**2)
            if dist < min:
                min = dist
                approx = last_xs[i]
        return approx, min == 0

    def get_delta_for_last_x(self, solu, x):
        """
        Gets the delta value for the last x.

        Parameters:
        - solu: The solu object.
        - x: The x value.

        Returns:
        - The delta value.
        """
        U, C = solu.sol(x)
        if C <= 0 or U > 1 or U <= 0:
            return -1
        delta = C**2 - (1 - U)**2
        return delta

    def sonic_U(self, delt):
        if self.omega <= 2:
            return 1/(self.gamma + 1)
        if self.omega > 2 and self.omega <= 3.26:
            return 1
        x_space = np.linspace(0, 1, self.precision)
        num_sol = solve_PDE(self.omega, delt, indi=False, x_end = 0).sol(x_space)
        U = num_sol[0].T
        C = num_sol[1].T
        delta = C**2 - (1 - U)**2
        negetive = np.where(delta <= 0)
        if len(negetive[0]) == 0:
            min_index = np.argmin(delta)
        else:
            min_index = negetive[0][-1]
        return U[min_index]

    def fast_last_x(self):
        if not self.last_x is None:
            return self.last_x
        U, C = self.get_UC()
        for i in range(len(U)):
            if self.omega <= 2:
                if U[i] <= 1/self.gamma:
                    print("touched U = ", 1/self.gamma, ", xi = ", self.xi[i])
                    return self.xi[i]
            if U[i]**2 + C[i]**2 > 10:
                print("escaped radius 10, last location - is U = ", U[i-1], ", C is ", C[i], "xi is ", self.xi[i-1])
                return self.xi[i]
            if C[i]**2 - (1 - U[i])**2 <= 0:
                print("crossed sonic line at - U = ", U[i], "C = ", C[i], "xi = ", self.xi[i])
                return self.xi[i]
        return 1


    def singuler_U(self, delt):
        """
        Calculates the singular U value.

        Returns:
        - The singular U value.
        """
        HH = (self.omega - 2*delt)/self.gamma
        if self.omega <= 3.26:
            sign = 1
        else:
            sign = -1
        return (delt + 2 + HH + sign * np.sqrt((delt + 2 + HH)**2 - 8 * HH)) / 4

    def plot(self, x_space=np.linspace(1, 0, sample_rate)):
        """
        Plots the U-C space.

        Parameters:
        - x_space: The x space array.
        """
        x_space = x_space**2
        line = np.linspace(1, -1, sample_rate)
        plt.plot(line, 1 - line, label="sonic line", color="green")
        plt.plot(line, line - 1, color="green")
        plt.plot(2/(self.gamma + 1), np.sqrt(2 * self.gamma * (self.gamma - 1)) / (self.gamma + 1), label="shock", marker="*", color="black")
        UU_1 = np.linspace(0.6001, 1, sample_rate)
        CC_1 = np.sqrt((self.gamma * (self.gamma - 1) * (1 - UU_1) * (UU_1 ** 2))/(2 * (self.gamma * UU_1 - 1)))
        plt.plot(UU_1, CC_1, label='first kind', color='black')
        if self.U is None or self.C is None:
            U_num, C_num = self.get_UC(x_space)
        else:
            U_num = self.U
            C_num = self.C
        plt.plot(U_num, C_num, marker='.', label="omega = " + str(self.omega))
        plt.xlabel("U")
        plt.ylabel("C")
        plt.xlim([-0.1, 1])
        plt.ylim([0, 1])
        plt.legend()
        plt.grid()
        plt.title("U-C Space")
        plt.show()

    def get_G(self, xi=np.linspace(1, 0, sample_rate)):
        """
        Gets the G array.

        Parameters:
        - xi: The xi array.

        Returns:
        - The G array.
        """
        if self.G is not None:
            return self.G
        lambd = (2*self.delt + self.omega * (self.gamma - 1))/(3 - self.omega)
        const = (((1 - 2/(self.gamma + 1)) ** lambd) * (((self.gamma + 1)/(self.gamma - 1)) ** (self.gamma + 1 + lambd)) )/(2 * self.gamma * (self.gamma - 1))
        U, C = self.get_UC(xi)
        G = (const * (C ** 2) * (xi ** (2 - 3 * lambd)) * ((1 - U) ** (-lambd))) ** (1/(self.gamma - 1 + lambd))
        self.G = G
        return G 

    def get_P(self):
        """
        Gets the P array.

        Returns:
        - The P array.
        """
        if self.P is not None:
            return self.P
        self.P = (self.xi ** 2) * self.G * (self.C ** 2)/self.gamma
        return self.P

    def get_derivs(self):
        """
        Gets the U, G, and P derivative arrays.

        Returns:
        - The U, G, and P derivative arrays.
        """
        if self.U_deriv is not None and self.C_deriv is not None and self.G_deriv is not None and self.P_deriv is not None:
            return self.U_deriv, self.G_deriv, self.P_deriv
        U, C = self.get_UC()
        G = self.get_G()
        P = self.get_P()
        self.U_deriv = np.gradient(U, self.xi)
        self.C_deriv = np.gradient(C, self.xi)
        self.G_deriv = np.gradient(G, self.xi)
        self.P_deriv = np.gradient(P, self.xi)
        return self.U_deriv, self.G_deriv, self.P_deriv

    def get_MM(self):
        """
        computes the MM inverse array.
        Returns:
        - The MM inverse array.
        """
        if self.MM is not None:
            return self.MM
        if self.last_x is None:
            self.last_x = self.last_X()  
        xi_begin = int(self.precision * (1 - self.last_x))
        U, C = self.get_UC()
        G = self.get_G()
        P = self.get_P()
        print("shape of U is ", U.shape, "shape of C is ", C.shape, "shape of G is ", G.shape, "shape of P is ", P.shape)
        xi = self.xi
        zeros = np.zeros(self.precision)
        M1 = np.array([xi * (U - 1), xi * G, zeros, zeros])
        M2 = np.array([zeros, (xi ** 2) * G * (U - 1), zeros, np.ones(self.precision)])
        M3 = np.array([zeros, zeros, (xi ** 2) * G * (U - 1), zeros])
        M4 = np.array([self.gamma * xi * (U - 1)/G, zeros, zeros, xi * (U - 1)/P])
        MM = np.stack((M1, M2, M3, M4))
        MM = np.rollaxis(MM, 2, 0)[:xi_begin+1]
        MM_inv = np.linalg.inv(MM)
        self.MM = MM
        self.MM_inv = MM_inv
        return MM
    

    def get_W0(self):
        """
        Computes the W0 matrix.
        """
        
        if self.W0 is not None:
            return self.W0
        
        if self.last_x is None:
            self.last_x = self.last_X()
        xi_begin = int(self.precision * (1 - self.last_x))
        print("xi_begin is ", xi_begin)
        
        xi = self.xi
        omega = self.omega
        U, C = self.get_UC()
        G = self.get_G()
        P = self.get_P()
        U_d, G_d, P_d = self.get_derivs()
        if self.MM is None:
            self.get_MM()



        zeros = np.zeros(self.precision)
        ones = np.ones(self.precision)

        N1 = np.array([omega - 3*U - xi * U_d, 
                       -xi*G_d - 3 * G, 
                       zeros, 
                       zeros])
        N2 = np.array([P_d/G, 
                       xi * G * (1 - self.delt - 2*U - xi * U_d), 
                       zeros, zeros])
        N3 = np.array([zeros, zeros, 
                       xi * G * (1 - self.delt - 2*U), 
                       -1/xi])
        N4 = np.array([- self.gamma * xi * (U - 1) * G_d/(G ** 2),
                        xi * self.gamma * G_d/G,
                        zeros, 
                        xi * (U - 1) * P_d/(P**2)])

        N0 = np.stack((N1, N2, N3, N4))
        N0 = np.rollaxis(N0, 2, 0)[:xi_begin+1]
        N0 = np.matmul(self.MM_inv, N0)

        print("N0 last is - ", N0[-1])

        # now we integrate the N0 matrix along the xi axis

        W0 = np.sum(N0, axis=0)/ (self.precision - xi_begin)
        self.W0 = W0
        print("W0 is ", W0)
        return W0



    def get_Wq(self):
        """
        Computes the Wq matrix.
        """
        
        if self.Wq is not None:
            return self.Wq
        
        if self.last_x is None:
            self.last_x = self.last_X()
        xi_begin = int(self.precision * (1 - self.last_x))
        print("xi_begin is ", xi_begin)
        
        xi = self.xi
        omega = self.omega
        U, C = self.get_UC()
        G = self.get_G()
        P = self.get_P()
        U_d, G_d, P_d = self.get_derivs()

        if self.MM is None:
            self.get_MM()


        zeros = np.zeros(self.precision)
        ones = np.ones(self.precision)

        N1 = np.array([-ones, 
                       zeros, 
                       zeros, 
                       zeros])
        N2 = np.array([zeros, 
                       - xi * G, 
                       zeros, zeros])
        N3 = np.array([zeros, zeros, 
                       -xi * G, 
                       zeros])
        N4 = np.array([self.gamma*ones / G,
                        zeros, zeros, 
                        -1/P])

        Nq = np.stack((N1, N2, N3, N4))
        Nq = np.rollaxis(Nq, 2, 0)[:xi_begin+1]
        Nq = np.matmul(self.MM_inv, Nq)

        print("Nq last is - ", Nq[-1])

        # now we integrate the N0 matrix along the xi axis

        Wq = np.sum(Nq, axis=0)/ (self.precision - xi_begin)
        self.Wq = Wq
        print("Wq is ", Wq)
        return Wq
    
    def get_Wl(self):
        """
        Computes the Wl matrix.
        """
        
        if self.Wl is not None:
            return self.Wl
        
        if self.last_x is None:
            self.last_x = self.last_X()
        xi_begin = int(self.precision * (1 - self.last_x))
        print("xi_begin is ", xi_begin)
        
        xi = self.xi
        omega = self.omega
        U, C = self.get_UC()
        G = self.get_G()
        P = self.get_P()
        U_d, G_d, P_d = self.get_derivs()

        if self.MM is None:
            self.get_MM()


        zeros = np.zeros(self.precision)
        ones = np.ones(self.precision)

        N1 = np.array([zeros, zeros, G, zeros])
        N2 = np.array([zeros, zeros, zeros, zeros])
        N3 = np.array([zeros, zeros, zeros ,zeros])
        N4 = np.array([zeros, zeros, zeros, zeros])

        Nl = np.stack((N1, N2, N3, N4))
        Nl = np.rollaxis(Nl, 2, 0)[:xi_begin+1]
        Nl = np.matmul(self.MM_inv, Nl)



        print("Nl last is", Nl[-1])
        # now we integrate the Wl matrix along the xi axis

        Wl = np.sum(Nl, axis=0)
        print("Wl before normalization is ", Wl)
        Wl = Wl / (self.precision - xi_begin)
        self.Wl = Wl
        print("Wl is ", Wl)
        return Wl

    def pertubation_BC(self, q):
        """
        Computes the pertubation boundary condition.
        """
        if self.P_deriv is None or self.U_deriv is None or self.G_deriv is None:
            self.get_derivs()

        P_d = - self.P_deriv[0]
        U_d = - self.U_deriv[0]
        G_d = - self.G_deriv[0]

        Y0 = - self.omega * (self.gamma + 1)/(self.gamma - 1) - G_d
        Y1 = 2 * q / (self.gamma + 1) - U_d
        Y2 = - 2 / (self.gamma + 1)
        Y3 = 2 * ( 2 * (q + 1) - self.omega) - P_d

        Y = np.array([Y0, Y1, Y2, Y3])
        return Y



