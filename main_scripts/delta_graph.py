import sys
sys.path.insert(0, 'C:\\projects\\repo\\strong_explosion_stability\\liberies')

from solution import solution

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


omegas = []
deltas = []


omega_range = [3,6]
precision = 20

omegas = np.linspace(omega_range[0], omega_range[1], precision)
deltas = []
for omega in omegas:
    sol = solution(omega)
    deltas.append(sol.delt)

plt.plot(omegas, deltas, '.')
plt.xlabel('omega')
plt.ylabel('delta')
plt.title('delta as a function of omega')
plt.grid()
plt.show()

