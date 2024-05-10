import sys
sys.path.insert(0, 'C:\\projects\\repo\\strong_explosion_stability\\liberies')

from solution import solution

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize


omegas = np.linspace(0, 7, 40)

deltas = [-1.49999653, -1.41025303, -1.32050772, -1.2307669, -1.14102521, -1.0512796,
        -0.96153606, -0.87179343, -0.78205128, -0.69230453, -0.60256112, -0.51282024,
        -0.42320911, -0.33564886, -0.25899203, -0.18158988, -0.11649657, -0.05075011,
        -0.00629744, 0.04115695, 0.08794111, 0.13342761, 0.17773719, 0.22020372,
        0.26359507, 0.30541043, 0.34670696, 0.38441937, 0.427785, 0.46782507,
        0.5073433, 0.54684478, 0.5860197, 0.62084639, 0.6636779, 0.70269384,
        0.74096621, 0.77984253, 0.81735509, 0.85564096]
"""
second_omegas = np.linspace(-2.1, 0, 6)

forth_deltas = np.zeros(len(second_omegas))

for i in range(len(second_omegas)):
    max_delta = scipy.optimize.minimize_scalar(
        lambda delt: solution(second_omegas[i], delt).fast_last_x(),  
        method='bounded',
        bracket=[-0.1,1.1],
        bounds=(-5, 0)
    )
    forth_deltas[i] = max_delta.x
    print("the max delta is -- " ,max_delta.x)

print(forth_deltas)
"""

third_omegas = [3, 3.26]
third_deltas = [0, 0]

first_omegas = [0, 3]
first_deltas = [-1.5, 0]

plt.plot(third_omegas, third_deltas, 'r')
plt.plot(first_omegas, first_deltas, 'g')
plt.plot(omegas, deltas, 'b.-')
plt.xlabel('omega')
plt.ylabel('delta')
plt.title('delta as a function of omega')
plt.grid()
plt.legend(['third type', 'first type', 'extended second type'])
plt.show()