import sys

import scipy.optimize
sys.path.insert(0, 'C:\\projects\\repo\\strong_explosion_stability\\liberies')

from solution import solution

import matplotlib.pyplot as plt
import numpy as np
import scipy


A = 1


fig, axs = plt.subplots(2, 2, figsize=(10, 8))

q_real = np.linspace(-1, 0, 100)
q_imagenry = np.linspace(-1, 1, 100)
qs = 
omegas = np.linspace(3, 3.25, 20)
for omega in omegas:
    sol = solution(omega, 0)
    last_x = sol.last_X()


    lambda_1 = []
    lambda_2 = []
    lambda_3 = []
    lambda_4 = []

    for q in qs:
        L1 = A * last_x/q
        L2 = A * last_x/(q + 1 -3*last_x*(1 - omega/5))
        L3 = A * last_x/(q + omega*(last_x/5 - 1) + 3*(1 - last_x))

        L4 = (omega * (1 - last_x/5) + 3*(last_x - 1) - q) * ((1+q)/last_x - 3*(1-omega/5))/((3 - omega * 6/5) * 2 * omega * (1 + q)/(5*A*last_x))

        lambda_1.append(L1)
        lambda_2.append(L2)
        lambda_3.append(L3)
        lambda_4.append(L4)



    axs[0, 0].plot(qs, lambda_1, label= "omega =" + str(omega))
    axs[0, 0].set_title('Lambda 1')
    axs[0, 0].grid()

    axs[0, 1].plot(qs, lambda_2, label= "omega =" + str(omega))
    axs[0, 1].set_title('Lambda 2')
    axs[0, 1].grid()

    axs[1, 0].plot(qs, lambda_3, label= "omega =" + str(omega))
    axs[1, 0].set_title('Lambda 3')
    axs[1, 0].grid()

    axs[1, 1].plot(qs, lambda_4, label= "omega =" + str(omega))
    axs[1, 1].set_title('Lambda 4')
    axs[1, 1].grid()

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()