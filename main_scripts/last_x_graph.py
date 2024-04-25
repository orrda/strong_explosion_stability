import sys
sys.path.insert(0, 'C:\\projects\\repo\\strong_explosion_stability\\liberies')

from solution import solution

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

omegas = np.linspace(2.00, 3.25, 100)
last_xs = []

for omega in omegas:
    sol = solution(omega)
    last_xs.append(sol.last_x)

plt.plot(omegas, last_xs, '.')
plt.xlabel('delta')
plt.ylabel('last_x')
plt.title('last_x as a function of delta')
plt.grid()
plt.show()


