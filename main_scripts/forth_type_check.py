import sys
sys.path.insert(0, 'C:\\projects\\repo\\strong_explosion_stability\\liberies')

from solution import solution

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize

sample_rate = 1000000

gamma = 5/3

omegas = [4.25]
colors = ['red', 'blue', 'green', 'black', 'yellow']

def get_max_delta(omega):
    max_delta = scipy.optimize.minimize_scalar(
        lambda delt: solution(omega, delt).fast_last_x(),  
        method='bounded',
        bracket=[-0.1,1.1],
        bounds=(0, 0.5),
        tol=1e-30
    )
    delta = max_delta.x
    last_x = solution(omega, delta).fast_last_x()
    return delta, last_x

deltas = [0,0.06,0.125,0.2,0.25]
omega = 4.25

for i in range(len(deltas)):

    sol = solution(omega, deltas[i])

    U_forth, C_forth = sol.get_UC(np.linspace(0, 1, sample_rate))

    plt.plot(U_forth[0], C_forth[0], marker = 'o', color = colors[i])
    plt.plot(U_forth, C_forth, color = colors[i], label="omega = " + str(omega))


x_space = np.linspace(1, 0, sample_rate)**2
line = np.linspace(1, -1, sample_rate)
plt.plot(line, 1 - line, label="sonic line", color="green")
plt.plot(line, line - 1, color="green")
plt.plot(2/(gamma + 1), np.sqrt(2 * gamma * (gamma - 1)) / (gamma + 1), label="shock", marker="*", color="black")
UU_1 = np.linspace(0.6001, 1, sample_rate)
CC_1 = np.sqrt((gamma * (gamma - 1) * (1 - UU_1) * (UU_1 ** 2))/(2 * (gamma * UU_1 - 1)))
plt.plot(UU_1, CC_1, label='first kind', color='black')

plt.xlabel("U")
plt.ylabel("C")
plt.xlim([-0.1, 1.1])
plt.ylim([0, 1])
plt.legend()
plt.grid()
plt.title("U-C Space")
plt.show()
