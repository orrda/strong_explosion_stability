import sys
sys.path.insert(0, 'C:\\projects\\repo\\strong_explosion_stability\\liberies')

from solution_int import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


omega_range = [0,7]
precision = 600

omegas = np.linspace(omega_range[0], omega_range[1], precision)
deltas = []
for omega in omegas:
    sol = solution(omega)
    delt = sol.find_delta()
    deltas.append(delt)




plt.plot(omegas, deltas, '.')
plt.xlabel('omega')
plt.ylabel('delta')
plt.title('delta as a function of omega')
plt.grid()
plt.show()
