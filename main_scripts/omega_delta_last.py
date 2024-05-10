import sys
sys.path.insert(0, 'C:\\projects\\repo\\strong_explosion_stability\\liberies')

from solution import solution

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize


"""
omegas = np.linspace(2.00, 7, 50)

deltas_len = 10


last_xs = np.zeros((len(omegas), deltas_len))

for i, omega in enumerate(omegas):
    sol = solution(omega, delt = 0.5)
    delta_inf, delta_sup = sol.check_DB(omega)
    approx = (delta_inf + delta_sup)/2
    deltas = np.linspace(delta_inf - 0.2, delta_sup + 0.2, deltas_len)
    for j, delt in enumerate(deltas):
        sol = solution(omega, delt)
        sol.last_x = sol.fast_last_x()
        if sol.last_x is not None:
            last_xs[i, j] = sol.last_x
            print(f'omega: {omega}, delt: {delt}, last_x: {sol.last_x}')
"""


path = "DB\\omega_delta.npy"
data = np.load(path)
omegas_w = data[:,0]
deltas_w = data[:,1]


omegas = np.linspace(2.00, 7, 40)
max_deltas = np.zeros(len(omegas))
for i in range(len(omegas)):
    max_delta = scipy.optimize.minimize_scalar(
        lambda delt: solution(omegas[i], delt).fast_last_x(),  
        method='bounded',
        bracket=[-0.6, 1.5],
        bounds=(-1, 1)
    )
    max_deltas[i] = max_delta.x
    print("the max delta is -- " ,max_delta.x)

print(max_deltas)

plt.plot(omegas, max_deltas)
plt.plot(omegas_w, deltas_w, '.')
plt.xlabel('omega')
plt.ylabel('delta')
plt.title('delta as a function of omega')
plt.grid()
plt.show()



