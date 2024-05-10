import sys

import scipy.optimize
sys.path.insert(0, 'C:\\projects\\repo\\strong_explosion_stability\\liberies')

from solution import solution

import matplotlib.pyplot as plt
import numpy as np
import scipy

omegas = np.linspace(2.1, 3, 7)
maxi_deltas = []

def get_max_delta(omega):
    max_delta = scipy.optimize.minimize_scalar(
        lambda delt: solution(omega, delt).fast_last_x(),  
        method='bounded',
        bracket=[-0.1,1.1],
        bounds=(-0.51, 0.01)
    )
    return max_delta.x


maxi_deltas = []
for omega in omegas:
    first_delta = (omega - 3) / 2
    max_delta = get_max_delta(omega)
    deltas = np.linspace(2 * max_delta - first_delta, 2 * first_delta - max_delta, 50)
    for delta in deltas:
        sol = solution(omega, delta)
        sol.last_x = sol.fast_last_x()
        if sol.last_x is not None:
            maxi_deltas.append([omega, delta, sol.last_x])
            print(f'omega: {omega}, delt: {delta}, last_x: {sol.last_x}')


maxi_deltas = np.array(maxi_deltas)
omega_values = maxi_deltas[:, 0]
delta_values = maxi_deltas[:, 1]
last_x_values = maxi_deltas[:, 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(omega_values, delta_values, last_x_values)
ax.set_xlabel('Omega')
ax.set_ylabel('Delta')
ax.set_zlabel('Last X')
plt.show()