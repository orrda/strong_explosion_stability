import numpy as np
import matplotlib.pyplot as plt


def delta0(U, C):
	return C ** 2 - (1 - U) ** 2


def delta1(U, C, omega, delt, gamma):
    return U * (1 - U) * (1 - U - delt) - (C ** 2) * (3 * U +(- omega + 2 * delt)/gamma)


def delta2(U, C, omega, delt, gamma):
	return C * (1 - U) * (1 - U - delt) - (gamma - 1) * C * U * (2 - 2 * U + delt) / 2 - (C ** 3) + (2 * delt + (gamma - 1) * omega) * (C ** 3)/(2 * gamma * (1 - U))


def ode_sys_by_t(t, y, args):
    omega, delt, gamma = args

    dU_dt = delta1(y[0], y[1], omega, delt, gamma)
    dC_dt = delta2(y[0], y[1], omega, delt, gamma)
    dXi_dt = delta0(y[0], y[1]) * y[2]

    return np.stack([dU_dt, dC_dt, dXi_dt])


def event_func(U, C):
    return np.min([C + U - 1.0, C, U, 1 - U])


def solveODE(func, y0, t, B_table, args=None):
    if args is None:
        args = ()
        
    c, A, b = B_table
    stages = len(b)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        k = np.zeros((stages, len(y0)))
        
        for s in range(stages):
            t_s = t[i] + c[s] * dt
            # Use dot product for explicit RK (A is strictly lower triangular)
            y_s = y[i] + dt * np.tensordot(A[s, :s], k[:s], axes=([0], [0]))
            k[s] = func(t_s, y_s, args)

        y[i+1] = y[i] + dt * np.tensordot(b, k, axes=([0], [0]))
        if event_func(y[i+1][0], y[i+1][1]) <= 0:
            print(f"Event triggered at t={t[i+1]:.8f}, U={y[i+1][0]:.8f}, C={y[i+1][1]:.8f}")
            return y[:i+1]

    return y

def Y0(gamma):
    U_init = 2/(gamma + 1)
    C_init = np.sqrt(2 * gamma * (gamma - 1)) / (gamma + 1)
    Xi_init = 1.0

    return np.array([U_init, C_init, Xi_init])


RK4_table = (
    np.array([0, 0.5, 0.5, 1]),
    np.array([[0, 0, 0, 0],
              [0.5, 0, 0, 0],
              [0, 0.5, 0, 0],
              [0, 0, 1, 0]]),
    np.array([1/6, 1/3, 1/3, 1/6])
)

DOPRI8_table = (
    np.array([0, 1/3, 2/5, 1, 2/3, 4/5]),
        np.array([[0, 0, 0, 0, 0, 0],
                  [1/3, 0, 0, 0, 0, 0],
                  [4/25, 6/25, 0, 0, 0, 0],
                  [1/4, -3, 15/4, 0, 0, 0],
                  [2/27, 10/9, -50/81, 8/81, 0, 0],
                  [2/25, 12/25, 2/15, 8/75, 0, 0]]),
        np.array([23/192, 0, 125/192, 0, -27/64, 125/192])
    )




if __name__ == "__main__":
    t = np.linspace(0, -510, 10000)
    y0 = Y0(5/3)

    y = solveODE(ode_sys_by_t, y0, t, RK4_table, args=(3.2554, 0, 5/3))


    U = y[:, 0]
    C = y[:, 1]

    plt.plot(U, C, ".")
    plt.xlabel("U")
    plt.ylabel("C")
    plt.xlim(0.6, 1.0)
    plt.ylim(0.0, 0.6)
    plt.grid()
    plt.title("Phase Space")
    plt.show()
