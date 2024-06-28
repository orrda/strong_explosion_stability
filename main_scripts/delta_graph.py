import sys
sys.path.insert(0, 'C:\\projects\\repo\\strong_explosion_stability\\liberies')

from solution import solution

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


omega_range = [3.35,6.05]
precision = 7

omegas = np.linspace(omega_range[0], omega_range[1], precision)
deltas = []
for omega in omegas:
    sol = solution(omega)
    U, C = sol.get_UC()
    plt.plot(U, C, label = f'omega = {omega}')


plt.xlabel('U')
plt.ylabel('C')
plt.title('U as a function of C')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.show()


path = "DB\\omega_delta.npy"
data = np.load(path).T

omegas = data[0]
deltas = data[1]



plt.plot(omegas, deltas, '.')
plt.xlabel('omega')
plt.ylabel('delta')
plt.title('delta as a function of omega')
plt.grid()
plt.show()
