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




sample_rate = 2000
omega = 4.25
delt = 0.25
sol = solution(omega=omega, delt=delt, sample_rate=sample_rate)
last_xi = sol.last_X()
max_dist = int(sample_rate - last_xi * sample_rate)
print(f"max_dist: {max_dist}")
max_dist = sample_rate



L = 0.5
per = perubation(sol, L)

res = 600


q_arr = np.linspace(-2, 2, res)

results = np.zeros((len(q_arr), max_dist))

for i, q_guess in enumerate(q_arr):
    if i % 10 == 0:
        print(f"Processing q_guess: {q_guess}")
    q_i = per.pressure_space(q_guess, max_dist)
    result = np.log(np.abs(q_i))
    result = np.where(result > 20, 10, result)
    results[i, :len(q_i)] = result

plt.figure(figsize=(10, 6))
plt.imshow(results, extent=[1, max_dist, q_arr[0], q_arr[-1]],
           aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Log(|DMdq|)')
plt.xlabel('Distance')
plt.ylabel('q')
plt.title('Log(|DMdq|) with L = ' + str(L))
plt.show()