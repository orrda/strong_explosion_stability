import numpy as np
from scipy.integrate import solve_ivp

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