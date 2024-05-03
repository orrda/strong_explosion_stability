import sys

import scipy.optimize
sys.path.insert(0, 'C:\\projects\\repo\\strong_explosion_stability\\liberies')

from solution import solution

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

omegas = np.linspace(2.00, 6, 30)
maxi_deltas = []

def func(omega):
    sol = solution(omega, delt = 0)
    delta_inf, delta_sup = sol.check_DB(omega)
    approx = (delta_inf + delta_sup)/2
    maximum = scipy.optimize.newton(
        lambda delt: sol.last_X,
        x0=approx,
        fprime = False,
        maxiter=500,
        disp=False
    )


    return sol.last_x

for omega in omegas:


    

plt.plot(omegas, last_xs, '.')
plt.xlabel('delta')
plt.ylabel('last_x')
plt.title('last_x as a function of delta')
plt.grid()
plt.show()
