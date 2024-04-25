import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

gamma = 5/3
n = 2
methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']

x_begin = 1
x_end = 1/gamma 
x_nsamples = 1000
x_space = np.linspace(x_begin, x_end, x_nsamples)

U_init = 2/(gamma + 1)
C_init = np.sqrt(2 * gamma * (gamma - 1)) / (gamma + 1)



def get_last(omega):
    event = [eventU, eventC]
    event[0].terminal = True
    event[1].terminal = True

    num_sol = solve_ivp(
        ode_sys, 
        [x_begin, x_end], 
        [U_init, C_init], 
        method=methods[0], 
        dense_output=True,
        events = event
    )

    last = num_sol.t_events[0]

    if len(last) == 0:
        last = x_end
    else:
        last = last[0]
    
    return last


def get_UC(omega):

    event = [eventU, eventC]
    event[0].terminal = True
    event[1].terminal = True

    num_sol = solve_ivp(
        ode_sys, 
        [x_begin, x_end], 
        [U_init, C_init], 
        method=methods[0], 
        dense_output=True,
        events = event
    )

    last = num_sol.t_events[0]

    if len(last) == 0:
        last = x_end
    else:
        last = last[0]
    print("last is ", last)
    x_space = np.linspace(x_begin, last, x_nsamples)

    UC_num_sol = num_sol.sol(x_space)
    U_num_sol = UC_num_sol[0].T
    C_num_sol = UC_num_sol[1].T

    return U_num_sol, C_num_sol

def delta(U, C):
	return C ** 2 - (1 - U) ** 2

def delta1(U, C):
    if omega < 3:
        lambd = (5 - omega)/2
    else:
        lambd = 1
    
    return (((omega + 2 * (lambd - 1))/gamma) - (n + 1) * U) * C ** 2 + U * (1 - U) * (lambd - U)

def delta2(U, C):
    if omega < 3:
        lambd = (5 - omega)/2
    else:
        lambd = 1
    
    return ((1 - U)**2 - n * (gamma - 1) * U * (1 - U)/2 + (lambd - 1) * (1 + (gamma - 3) * U/2)) * C - (1 + ((2 * (lambd - 1) - (gamma - 1) * omega))/(2 * gamma * (1 - U))) * C**3

def ode_sys(x, UC):
    
    if omega < 3:
        lambd = (5 - omega)/2
    else:
        lambd = 1
    

    U=UC[0]
    C=UC[1]
    dU_dx = delta1(U, C)/(delta(U, C) * lambd * x)
    dC_dx = delta2(U, C)/(delta(U, C) * lambd * x)
    return [dU_dx, dC_dx]


def eventU(x, UC):
    U=UC[0]
    return U - 1

def eventC(x, UC):
    C=UC[1]
    return C



line = np.linspace(0, 1, 1000)


plt.show()

xi_vacum = []

omegas2 = np.linspace(2, 3, 100)
omegas3 = np.linspace(3, 3.26, 10)

omegas = np.concatenate((omegas2, omegas3))

for omega in omegas:
    last = get_last(omega)
    if last == 0:
        last = None
    else:
        last = last
    xi_vacum.append(last)

plt.title('Vacum by Omega')
plt.plot(omegas, xi_vacum,marker = '.', color='black')
plt.xlabel('Omega')
plt.ylabel('X vacum')

plt.plot(3 * np.ones(100),np.linspace(0, 1, 100), color='red')
plt.plot(3.26 * np.ones(100),np.linspace(0, 1, 100), color='red')
plt.grid()
plt.show()

for omega in omegas3:
    print("Omega is - " + str(omega)[:5])

    plt.plot(line, 1-line)
    plt.plot(U_init, C_init, label='Shock', marker='*', color='black')
    plt.xlabel('U')
    plt.ylabel('C')

    UU_1 = np.linspace(1/gamma + 0.001, 1, 1000)
    CC_1 = np.sqrt((gamma * (gamma - 1) * (1 - UU_1) * (UU_1 ** 2))/(2 * (gamma * UU_1 - 1)))
    plt.plot(UU_1, CC_1, color='blue')

    U_num_sol, C_num_sol = get_UC(omega)
    U_num_sol = U_num_sol[:len(U_num_sol) - 2]
    C_num_sol = C_num_sol[:len(C_num_sol) - 2]
    plt.plot(U_num_sol, C_num_sol, marker = ".", label='U', color='red')
    plt.show()


plt.show()