import numpy as np
import matplotlib.pyplot as plt

from liberies.PDE import *

def find_delta(omega, gamma):
    if omega <= 3:
        return (omega - 3)/2
    if omega <= 3.2554:
        return 0.0
    
    delt_min = 0.0
    delt_max = 1.0

    while delt_max - delt_min > 1e-12:
        delt = (delt_min + delt_max) / 2
        if check_delta(omega, gamma, delt):
            delt_max = delt
        else:
            delt_min = delt

    return delt_min

def check_delta(omega, gamma, delt):

    HH = (omega - 2*delt)/gamma
    term_sqrt = np.maximum((delt + 2 + HH)**2 - 8 * HH, 0.0)
    sing_U = (delt + 2 + HH  - np.sqrt(term_sqrt)) / 4

    args = (omega, delt, gamma)
    t = np.linspace(0, -40, 5000)
    y0 = Y0(5/3)

    y = solveODE(ode_sys_by_t, y0, t, DOPRI8_table, args=args)
    U_last = y[-1, 0]

    return U_last > sing_U

