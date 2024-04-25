import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import scipy.optimize

from PDE import *

sample_rate = 1000

class solution:
    """
    Represents a solution to a PDE problem.

    Attributes:
    - omega: The value of omega.
    - delt: The value of delt.
    - gamma: The value of gamma.
    - last_x_approx: The last x approximation.
    - last_delta_approx: The last delta approximation.
    - xi: The array of xi values.
    - precision: The precision value.
    - last_x: The last x value.
    - U: The U array.
    - C: The C array.
    - G: The G array.
    - P: The P array.
    - U_deriv: The U derivative array.
    - C_deriv: The C derivative array.
    - G_deriv: The G derivative array.
    - P_deriv: The P derivative array.
    - MM: The MM array.
    - MM_inv: The inverse of MM array.

    Methods:
    - find_delta: Finds the value of delta.
    - find_delta_helper: Helper function for finding delta.
    - get_UC: Gets the U and C arrays.
    - last_X: Finds the last x value.
    - get_delta_for_last_x: Gets the delta value for the last x.
    - sonic_U: Calculates the sonic U value.
    - singuler_U: Calculates the singular U value.
    - plot: Plots the U-C space.
    - get_G: Gets the G array.
    - get_P: Gets the P array.
    - get_derivs: Gets the U, G, and P derivative arrays.
    - get_MM: Gets the MM inverse array.
    """

    def __init__(self, omega, delt=None, gamma=5/3):
        """
        Initializes a new instance of the solution class.

        Parameters:
        - omega: The value of omega.
        - delt: The value of delt.
        - gamma: The value of gamma.
        - last_x_approx: The last x approximation.
        - last_delta_approx: The last delta approximation.
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
            self.last_x = self.last_X()
        self.U_deriv = None
        self.C_deriv = None
        self.G_deriv = None
        self.P_deriv = None
        self.MM = None
        self.MM_inv = None

    def find_delta(self):
        """
        Finds the value of delta.

        Returns:
        - The value of delta.
        """
        if self.omega <= 3:
            self.delt = (self.omega - 3) / 2
        if self.omega > 3 and self.omega <= 3.26:
            self.delt = 0
        if self.omega > 3.26:
            delta_infimum, delta_suprimum = self.check_DB(self.omega)
            if delta_infimum == delta_suprimum:
                self.delt = delta_infimum
            else:
                func = lambda delt: self.find_delta_helper(delt)
                print("delta_infimum is ", delta_infimum, "delta_suprimum is ", delta_suprimum)
                print("func(inf)", func(delta_infimum), ", func(sup)", func(delta_suprimum))
                root = scipy.optimize.root_scalar(
                        self.find_delta_helper, 
                        method = 'brentq',
                        bracket = [delta_infimum,delta_suprimum],
                        maxiter=100
                    )
                print("for omega =", self.omega, ", delta =", root)
                self.delt = root.root
                
                self.save_DB(self.omega, self.delt)
        self.last_x = self.last_X()
        return self.delt

    def find_delta_helper(self, delt):
        """
        Helper function for finding delta.

        Parameters:
        - delt: The value of delt.

        Returns:
        - The difference between sonic_U and singuler_U.
        """
        sol = solution(self.omega, delt)
        diff = sol.sonic_U() - sol.singuler_U()
        print("for omega =", self.omega, ", delta =", delt, ", diff is - ", diff)
        return diff


    def check_DB(self,omega):
        path = "DB\\omega_delta.npy"
        data = np.load(path)

        print("data is ", data)
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
        """
        approx, is_there = self.check_last_x_DB()
        if is_there:
            return approx

        """
        
        solu = solve_PDE(self.omega, self.delt, gamma=self.gamma)
        x = scipy.optimize.root_scalar(
                lambda x: self.get_delta_for_last_x(solu, x),
                method='brentq',
                bracket=[0, 1],
                maxiter=100
            )
        self.last_x = x.root
        #self.save_last_x_DB(self.last_x)
        return self.last_x

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

    def sonic_U(self):
        """
        Calculates the sonic U value.

        Returns:
        - The sonic U value.
        """
        if self.omega <= 2:
            return 1/self.gamma
        elif self.omega <= 3.25 and self.omega >= 2:
            return 1
        else:
            last = self.last_x
            U, C = self.get_UC(x_space=np.linspace(1, last, sample_rate))
            return U[-1]

    def singuler_U(self):
        """
        Calculates the singular U value.

        Returns:
        - The singular U value.
        """
        HH = (self.omega - 2*self.delt)/self.gamma
        if self.omega <= 3.26:
            sign = 1
        else:
            sign = -1
        return (self.delt + 2 + HH + sign * np.sqrt((self.delt + 2 + HH)**2 - 8 * HH)) / 4

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
        Gets the MM inverse array.

        Returns:
        - The MM inverse array.
        """
        if self.MM is not None:
            return self.MM
        xi_begin = self.last_x
        print("xi_begin is ", xi_begin)    
        xi_begin = int(xi_begin * self.precision)
        U, C = self.get_UC()
        G = self.get_G()
        P = self.get_P()
        xi = self.xi
        zeros = np.zeros(self.precision)
        M1 = np.array([xi * (U - 1), xi * G, zeros, zeros])
        M2 = np.array([zeros, (xi ** 2) * G * (U - 1), zeros, np.ones(self.precision)])
        M3 = np.array([zeros, zeros, (xi ** 2) * G * (U - 1), zeros])
        M4 = np.array([self.gamma * xi * (U - 1)/G, zeros, zeros, xi * (U - 1)/P])
        MM = np.stack((M1, M2, M3, M4))
        MM = np.rollaxis(MM, 2, 0)[xi_begin:self.precision - 1]
        MM_inv = np.linalg.inv(MM)
        self.MM = MM
        self.MM_inv = MM_inv
        return MM_inv
