import numpy as np
from scipy.integrate import solve_ivp
import scipy.optimize
import matplotlib.pyplot as plt

def delta(U, C):
	return C ** 2 - (1 - U) ** 2

def delta1_xi(U, C, omega, delt, gamma):
    return U * (1 - U) * (1 - U - delt) - (C ** 2) * (3 * U +(- omega + 2 * delt)/gamma)

def delta2_xi(U, C, omega, delt, gamma):
	return C * (1 - U) * (1 - U - delt) - (gamma - 1) * C * U * (2 - 2 * U + delt) / 2 - (C ** 3) + (2 * delt + (gamma - 1) * omega) * (C ** 3)/(2 * gamma * (1 - U))


def ode_sys_by_xi(xi, UC, omega, delt, gamma):
    U=UC[0]
    C=UC[1]

    deltt = (delta(U, C) * xi)

    dU_dx = delta1_xi(U, C, omega, delt, gamma)/deltt
    dC_dx = delta2_xi(U, C, omega, delt, gamma)/deltt

    return [dU_dx, dC_dx]

def ode_sys_indi_xi(xi, UC, omega, delt, gamma):
    U=UC[0]
    C=UC[1]

    dU_dx = delta1_xi(U, C, omega, delt, gamma)
    dC_dx = delta2_xi(U, C, omega, delt, gamma)

    return [dU_dx, dC_dx]


def event_sonic(x, UC):
    return UC[1]**2 - (1 - UC[0])**2

def event_neg_C(x, UC):
    return UC[1]

def event_diverging(x, UC):
    return 10 - UC[0]**2 - UC[1]**2




def solve_PDE(omega, delt, indi = False,x_begin = 1, x_end = 0, gamma = 5/3):

    U_init = 2/(gamma + 1)
    C_init = np.sqrt(2 * gamma * (gamma - 1)) / (gamma + 1)

    event_sonic.terminal = True
    event_neg_C.terminal = True
    event_diverging.terminal = True

    if not indi:
        func = lambda x, UC: ode_sys_by_xi(x, UC, omega, delt, gamma)
    else:
        func = lambda x, UC: ode_sys_indi_xi(x, UC, omega, delt, gamma)

    num_sol = solve_ivp(
        func, 
        [x_begin, x_end], 
        [U_init, C_init], 
        method='RK45', 
        dense_output=True,
        events = [event_sonic, event_diverging, event_neg_C],
        max_step = 0.01
    )
    return num_sol


def sonic_U(omega, delt, gamma):
    x_space = np.linspace(0, 1, 10000)
    num_sol = solve_PDE(omega, delt, indi=False, x_end = 0).sol(x_space)
    U = num_sol[0].T
    C = num_sol[1].T
    delta = C**2 - (1 - U)**2
    negetive = np.where(delta <= 0)
    if len(negetive[0]) == 0:
        min_index = np.argmin(delta)
    else:
        min_index = negetive[0][-1]
    return U[min_index]

def UC_plot(U, C):
    gamma = 5/3
    plt.plot(U, C)
    line = np.linspace(1, -1, 1000)
    plt.plot(line, 1 - line, label="sonic line", color="green")
    plt.plot(line, line - 1, color="green")
    plt.plot(2/(gamma + 1), np.sqrt(2 * gamma * (gamma - 1)) / (gamma + 1), label="shock", marker="*", color="black")
    UU_1 = np.linspace(0.6001, 1, 1000)
    CC_1 = np.sqrt((gamma * (gamma - 1) * (1 - UU_1) * (UU_1 ** 2))/(2 * (gamma * UU_1 - 1)))
    plt.plot(UU_1, CC_1, label='first kind', color='black')
    plt.xlabel("U")
    plt.ylabel("C")
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.show()

def find_delta(omega, gamma, last_delta,upper_delta):
    func = lambda delt: abs(sonic_U(omega, delt, gamma) - singuler_U(omega, delt, gamma))
    optimze_delta = scipy.optimize.minimize_scalar(
            func,
            bounds=(last_delta, upper_delta),
            method='bounded',
            tol=1e-10
        )
    delta = optimze_delta.x
    print("omega is ", omega, "delta is ", delta, "error is ", func(delta))
    return delta



def singuler_U(omega, delt, gamma):
    HH = (omega - 2*delt)/gamma
    if omega <= 3.26:
        sign = 1
    else:
        sign = -1
    return (delt + 2 + HH + sign * np.sqrt((delt + 2 + HH)**2 - 8 * HH)) / 4


omegas = np.linspace(3.26, 5, 100)

deltas = []
last_delta = 0
for omega in omegas:
    upper_delta = last_delta + 0.05
    deltai = find_delta(omega, 5/3, last_delta, upper_delta)
    last_delta = deltai
    deltas.append(deltai)

plt.plot(omegas, deltas, '.')
plt.xlabel("omega")
plt.ylabel("delta")
plt.grid()
plt.show()








"""

x_space = np.linspace(0, 1, 1000)
for i in range(len(omegas)):
    U, C = solve_PDE(omegas[i], deltas[i], x_end = 0).sol(x_space)
    plt.plot(U, C)

gamma = 5/3
line = np.linspace(1, -1, 1000)
plt.plot(line, 1 - line, label="sonic line", color="green")
plt.plot(line, line - 1, color="green")
plt.plot(2/(gamma + 1), np.sqrt(2 * gamma * (gamma - 1)) / (gamma + 1), label="shock", marker="*", color="black")
UU_1 = np.linspace(0.6001, 1, 1000)
CC_1 = np.sqrt((gamma * (gamma - 1) * (1 - UU_1) * (UU_1 ** 2))/(2 * (gamma * UU_1 - 1)))
plt.plot(UU_1, CC_1, label='first kind', color='black')
plt.xlabel("U")
plt.ylabel("C")
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.show()

"""