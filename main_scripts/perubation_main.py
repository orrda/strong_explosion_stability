import sys
sys.path.insert(0, 'C:\\projects\\repo\\strong_explosion_stability\\liberies')

from perubation_class import perubation
from solution import solution

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize


sol = solution(omega = 4.25, delt = 0.25)

sol.plot()

L_array = np.linspace(0.05, 0.6, 50)

sonic_values = []

q_array = []
for L in L_array:
    per = perubation(sol, L)
    q = per.get_q()
    q_array.append(q)


plt.plot(L_array, q_array, '.-')
plt.xlabel("L")
plt.ylabel("q")
plt.title("q as a function of L, with omega = 4.25")
plt.grid()
plt.show()