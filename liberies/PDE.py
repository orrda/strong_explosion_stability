import numpy as np
from scipy.integrate import solve_ivp

def delta(U, C):
	return C ** 2 - (1 - U) ** 2

def delta1_x(U, C, omega, lambd, gamma):
    return (((omega + 2 * (lambd - 1))/gamma) - (2 + 1) * U) * C ** 2 + U * (1 - U) * (lambd - U)

def delta2_x(U, C, omega, lambd, gamma):
    return ((1 - U)**2 - 2 * (gamma - 1) * U * (1 - U)/2 + (lambd - 1) * (1 + (gamma - 3) * U/2)) * C - (1 + ((2 * (lambd - 1) - (gamma - 1) * omega))/(2 * gamma * (1 - U))) * C**3

def delta1_xi(U, C, omega, delt, gamma):
    return U * (1 - U) * (1 - U - delt) - (C ** 2) * (3 * U +(- omega + 2 * delt)/gamma)

def delta2_xi(U, C, omega, delt, gamma):
	return C * (1 - U) * (1 - U - delt) - (gamma - 1) * C * U * (2 - 2 * U + delt) / 2 - (C ** 3) + (2 * delt + (gamma - 1) * omega) * (C ** 3)/(2 * gamma * (1 - U))



def ode_sys_by_x(x, UC, omega, lambd, gamma):
    U=UC[0]
    C=UC[1]

    dU_dx = delta1_x(U, C, omega, lambd, gamma)/(delta(U, C) * lambd * x)
    dC_dx = delta2_x(U, C, omega, lambd, gamma)/(delta(U, C) * lambd * x)

    return [dU_dx, dC_dx]

def ode_sys_by_xi(xi, UC, omega, delt, gamma):
    U=UC[0]
    C=UC[1]

    deltt = (delta(U, C) * xi)
    #if abs(deltt) < 1e-20:
    #     deltt = 1e-20


    dU_dx = delta1_xi(U, C, omega, delt, gamma)/deltt
    dC_dx = delta2_xi(U, C, omega, delt, gamma)/deltt

    return [dU_dx, dC_dx]

def ode_sys_indi_xi(xi, UC, omega, delt, gamma):
    U=UC[0]
    C=UC[1]

    dU_dx = delta1_xi(U, C, omega, delt, gamma)
    dC_dx = delta2_xi(U, C, omega, delt, gamma)

    return [dU_dx, dC_dx]

def ode_sys_indi_x(x, UC, omega, lambd, gamma):
    U=UC[0]
    C=UC[1]

    dU_dx = delta1_x(U, C, omega, lambd, gamma)
    dC_dx = delta2_x(U, C, omega, lambd, gamma)

    return [dU_dx, dC_dx]


def event_sonic(x, UC):
    return UC[1]**2 - (1 - UC[0])**2

def event_neg_C(x, UC):
    return UC[1]

def event_diverging(x, UC):
    return 10 - UC[0]**2 - UC[1]**2




def solve_PDE(omega, delt, x_begin = 1, x_end = 0, gamma = 5/3):
    print("omega is ", omega, "delt is ", delt)

    U_init = 2/(gamma + 1)
    C_init = np.sqrt(2 * gamma * (gamma - 1)) / (gamma + 1)

    event_sonic.terminal = True
    event_neg_C.terminal = True
    event_diverging.terminal = True


    num_sol = solve_ivp(
        lambda x, UC: ode_sys_by_xi(x, UC, omega, delt, gamma), 
        [x_begin, x_end], 
        [U_init, C_init], 
        method='RK45', 
        dense_output=True,
        events = [event_sonic, event_diverging, event_neg_C],
        max_step = 0.001
    )

    return num_sol