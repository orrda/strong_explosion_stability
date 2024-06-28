import sys

import scipy.optimize
sys.path.insert(0, 'C:\\projects\\repo\\strong_explosion_stability\\liberies')

from solution import solution

import matplotlib.pyplot as plt
import numpy as np
import scipy

omegas = np.linspace(3, 3.25, 100)
last_xs = []

q2s = []
q3s = []

for omega in omegas:
    sol = solution(omega, 0)
    last_x = sol.last_X()
    if last_x is not None:
        last_xs.append(last_x)
        q2 = omega*(1 - last_x/5) + 3*last_x - 3
        q3 = 3*last_x - (3/5)*omega*last_x - 1
        q2s.append(q2)
        q3s.append(q3)

fig, ax = plt.subplots()
ax.plot(omegas, last_xs,".")
ax.set_xlabel('Omega')
ax.set_ylabel('Last X')
plt.show()

fig, ax = plt.subplots()
ax.plot(omegas, q2s, ".")
ax.plot(omegas, q3s, ".")
ax.set_xlabel('Omega')
ax.set_ylabel('Q')
plt.show()



