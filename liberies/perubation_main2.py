import numpy as np
import scipy.integrate
import scipy.linalg
from solution2 import *
from perubation_class2 import *
from PDE import *
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import root



omega_space = [4.25 ]
L_space = np.linspace(0.01, 1, 20)
for omega in omega_space:
    sol = solution(omega=4.25,delt=0.25)

    q1_space = []
    q2_space = []
    q_guess=-0.5
    for L in L_space:
        per = perubation(sol, L)
        q1 = per.get_q_new(q_guess)
        print(f"omega: {omega}, L: {L}, q1: {q1}")
        q1_space.append(q1)
        q_guess = q1

    plt.plot(L_space, q1_space, label=f'q, omega={omega:.2f}')

    

plt.xlabel("L")
plt.ylabel("q")
plt.title("q as a function of L")
plt.grid()
plt.legend()
plt.show()




