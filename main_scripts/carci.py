import sys
sys.path.insert(0, 'C:\\projects\\repo\\strong_explosion_stability\\liberies')

from solution import solution

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize

def delta1_xi(U, C, omega, delt, gamma):
    return U * (1 - U) * (1 - U - delt) - (C ** 2) * (3 * U +(- omega + 2 * delt)/gamma)

def delta2_xi(U, C, omega, delt, gamma):
	return C * (1 - U) * (1 - U - delt) - (gamma - 1) * C * U * (2 - 2 * U + delt) / 2 - (C ** 3) + (2 * delt + (gamma - 1) * omega) * (C ** 3)/(2 * gamma * (1 - U))


omega = 4.25
delatas = np.linspace(-0.6, 1.5, 40)

U_array = np.linspace(0, 1, 100)

C_array = np.linspace(0, 1, 100)

plt.figure()
for delt in delatas:
    gamma = 5/3
    xi1 = []
    xi2 = []
    for U in U_array:
        for C in C_array:
            xi1.append(delta1_xi(U, C, omega, delt, gamma))
            xi2.append(delta2_xi(U, C, omega, delt, gamma))
    xi2 = np.array(xi2)
    xi2 = xi2.reshape((len(U_array), len(C_array)))
    xi1 = np.array(xi1)
    xi1 = xi1.reshape((len(U_array), len(C_array)))
    plt.contour(U_array, C_array, xi1, levels=[0], colors='black')
    plt.contour(U_array, C_array, xi2, levels=[0], colors='blue')

plt.xlabel("U")
plt.ylabel("C")
plt.title("Delta 1")
plt.grid()
plt.show()


