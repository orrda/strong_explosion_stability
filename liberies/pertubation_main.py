import numpy as np
import scipy.integrate
import scipy.linalg
from solution_int import *
from PDE import *
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import root




sol = solution(omega=3.1)
delt = sol.find_delta()
print("delta is ", delt)

print(sol.xi)



W0 = sol.get_W0() 
Wq = sol.get_Wq()
Wl = sol.get_Wl()



def get_final_Y_norm(BC1, W0, Wq, Wl, q, l):
    mat = W0 + q * Wq + l * (l+1) * Wl
    mat = scipy.linalg.expm(-mat)
    Y = mat.dot(BC1)
    Y_norm = abs(Y[3])
    #print("q is ", q, " and l is ", l, "Y is ", Y, " and Y_norm is ", Y_norm)
    return Y_norm




l_space = np.linspace(0, 1, 50)
q_space = np.linspace(-1, 0, 50)
arr = np.zeros((len(l_space), len(q_space)))

for i, l in enumerate(l_space):
    for j, q in enumerate(q_space):
        Y_norm = get_final_Y_norm(sol.pertubation_BC(q), W0, Wq, Wl, q, l)
        arr[i, j] = Y_norm

plt.imshow(arr, extent=(-1, 0, 0, 1), aspect='auto', origin='lower', interpolation='nearest')
plt.colorbar(label='Y Norm')
plt.xlabel('q')
plt.ylabel('l')
plt.title('Y Norm as a function of q and l')
plt.show()
        



q_space = []
for l in l_space:
    q = scipy.optimize.minimize_scalar(
        lambda q: get_final_Y_norm(sol.pertubation_BC(q), W0, Wq, Wl, q, l),
        bounds =(-1, 0),
    ).x
    q_space.append(q)

q_space = np.array(q_space)
plt.plot(l_space, q_space, label='q vs l', marker = '.')
plt.xlabel('l')
plt.ylabel('q')
plt.title('q vs l')
plt.grid()
plt.legend()
plt.show()


