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


sample_rate = 1000
omega = 4.25
delt = 0.25
sol = solution(omega=omega, delt=delt, sample_rate=sample_rate)
last_xi = sol.last_X()
max_dist = int(sample_rate - last_xi * sample_rate)
print(f"max_dist: {max_dist}")

L = 0.05
per = perubation(sol, L)

res = 200


q_arr = np.linspace(-1, 1, res)
distances = np.linspace(1, 1000, res*2)

results = np.zeros((len(q_arr), len(distances)))


for i, q_guess in enumerate(q_arr):
    for j, distance in enumerate(distances):
        distance = int(distance)  # Ensure distance is an integer
        DMdq = per.get_DMDq(q_guess, distance)
        results[i, j] = np.log(np.abs(DMdq))

plt.figure(figsize=(10, 6))
plt.imshow(results, extent=[distances[0], distances[-1], q_arr[0], q_arr[-1]], 
           aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Log(|DMdq|)')
plt.xlabel('Distance')
plt.ylabel('q')
plt.title('Log(|DMdq|) as a function of Distance and q')
plt.show()

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




