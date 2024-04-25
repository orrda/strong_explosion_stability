import sys
sys.path.insert(0, 'C:\\projects\\repo\\strong_explosion_stability\\liberies')

from solution import solution

import matplotlib.pyplot as plt
import numpy as np


omegas = np.linspace(2.00, 4, 30)
deltas = np.linspace(-0.5, 0.5, 30)

last_xs = np.zeros((len(omegas), len(deltas)))

for i, omega in enumerate(omegas):
    for j, delt in enumerate(deltas):
        sol = solution(omega, delt)
        if sol.last_x is not None:
            last_xs[i, j] = sol.last_x

plt.imshow(last_xs, extent=[deltas[0], deltas[-1], omegas[0], omegas[-1]], aspect='auto')
plt.colorbar()
plt.xlabel('delta')
plt.ylabel('omega')
plt.title('last_x as a function of delta and omega')
plt.show()
