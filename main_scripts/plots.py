import sys

import scipy.optimize
sys.path.insert(0, 'C:\\projects\\repo\\strong_explosion_stability\\liberies')

from solution import solution

import matplotlib.pyplot as plt
import numpy as np
import scipy

omegas = np.linspace(0, 1, 9)
omegas = omegas * 0.253 + 3

for omega in omegas:
    sol = solution(omega, 0)
    U, C = sol.get_UC()
    plt.plot(U, C)
    sample_rate = sol.precision

"""
Plots the U-C space
Parameters:
- x_space: The x space array.
"""

gamma = 5/3

x_space=np.linspace(1, 0, sample_rate)
x_space = x_space**2
line = np.linspace(1, -1, sample_rate)
plt.plot(line, 1 - line, label="sonic line", color="green")
plt.plot(line, line - 1, color="green")
plt.plot(2/(gamma + 1), np.sqrt(2 * gamma * (gamma - 1)) / (gamma + 1), label="shock", marker="*", color="black")
UU_1 = np.linspace(0.6001, 1, sample_rate)
CC_1 = np.sqrt((gamma * (gamma - 1) * (1 - UU_1) * (UU_1 ** 2))/(2 * (gamma * UU_1 - 1)))
plt.plot(UU_1, CC_1, label='first kind', color='black')



plt.xlabel("U")
plt.ylabel("C")
plt.xlim([0.6, 1])
plt.ylim([0, 1])
plt.legend()
plt.grid()
plt.title("U-C Space")
plt.savefig("3rd_type_solutions.png", dpi=900)
plt.show()