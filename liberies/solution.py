from PDE import *
import numpy as np
import matplotlib.pyplot as plt

from delta_finder import find_delta



def G_val(U, C, xi, args):
    omega, delt, gamma = args
    lambd = (2 * delt + omega * (gamma - 1)) / (3 - omega)

    const = (((1 - 2/(gamma + 1)) ** lambd) * (((gamma + 1)/(gamma - 1)) ** (gamma + 1 + lambd)) )/(2 * gamma / (gamma - 1))
    G = (const * (C ** 2) * (xi ** (2 - 3 * lambd)) * ((1 - U) ** (-lambd))) ** (1/(gamma - 1 + lambd))
    return G


def P_val(U, C, xi, args):
    G = G_val(U, C, xi, args)

    P = (xi ** 2) * G * (C ** 2) / args[2]
    return P



def dUdxi_val(U, C, xi, args):
    omega, delt, gamma = args
    delta0 = C ** 2 - (1 - U) ** 2
    delta1 = U * (1 - U) * (1 - U - delt) - (C ** 2) * (3 * U + (-omega + 2 * delt) / gamma)

    return delta1/(delta0 * xi)


def dCdxi_val(U, C, xi, args):
    omega, delt, gamma = args
    delta0 = C ** 2 - (1 - U) ** 2
    delta1 = C * (1 - U) * (1 - U - delt) - (gamma - 1) * C * U * (2 - 2 * U + delt) / 2 - (C ** 3) + (2 * delt + (gamma - 1) * omega) * (C ** 3)/(2 * gamma * (1 - U))

    return delta1/(delta0 * xi)


def dGdxi_val(U, C, xi, args):
    omega, delt, gamma = args
    lambd = (2 * delt + omega * (gamma - 1)) / (3 - omega)
    G = G_val(U, C, xi, args)

    C_term = 2 * dCdxi_val(U, C, xi, args) / C
    U_term = lambd * dUdxi_val(U, C, xi, args) / (1 - U)
    xi_term = (2 - 3 * lambd) / xi

    sum_terms = C_term + U_term + xi_term
    return sum_terms * G / (gamma - 1 + lambd)


def dPdxi_val(U, C, xi, args):
    G = G_val(U, C, xi, args)
    P = P_val(U, C, xi, args)

    C_term = 2 * dCdxi_val(U, C, xi, args) / C
    G_term = dGdxi_val(U, C, xi, args) / G
    xi_term = 2 / xi

    return (C_term + G_term + xi_term) * P



def NNr(U, C, xi, args):
    omega, delt, gamma = args
    G = G_val(U, C, xi, args)
    P = P_val(U, C, xi, args)

    dUdxi = dUdxi_val(U, C, xi, args)
    dGdxi = dGdxi_val(U, C, xi, args)
    dPdxi = dPdxi_val(U, C, xi, args)

    zero_arr = np.zeros_like(xi)

    NN00 = omega - 3 * U - xi * dUdxi
    NN01 = - xi * dGdxi - 3 * G
    NN02 = zero_arr
    NN03 = zero_arr

    NN10 = dPdxi/G
    NN11 = (1 - delt - 2 * U - xi * dUdxi) * G * xi
    NN12 = zero_arr
    NN13 = zero_arr

    NN20 = zero_arr
    NN21 = zero_arr
    NN22 = (1 - delt - 2 * U) * G * xi
    NN23 = -1/xi

    NN30 = - gamma * (U - 1) * xi * dGdxi/(G**2)
    NN31 = gamma * xi * dGdxi/G
    NN32 = zero_arr
    NN33 = xi * (U - 1) * dPdxi/(P ** 2)

    return np.array([[NN00, NN01, NN02, NN03],
                        [NN10, NN11, NN12, NN13],
                        [NN20, NN21, NN22, NN23],
                        [NN30, NN31, NN32, NN33]])


def MMr(U, C, xi, args):
    omega, delt, gamma = args

    G = G_val(U, C, xi, args)
    P = P_val(U, C, xi, args)

    zero_arr = np.zeros_like(xi)

    MM00 = xi * (U - 1)
    MM01 = G * xi
    MM02 = zero_arr
    MM03 = zero_arr

    MM10 = zero_arr
    MM11 = (U - 1) * G * xi**2
    MM12 = zero_arr
    MM13 = np.ones_like(xi)

    MM20 = zero_arr
    MM21 = zero_arr
    MM22 = (U - 1) * G * xi**2
    MM23 = zero_arr

    MM30 = - gamma * (U - 1) * xi / G
    MM31 = zero_arr
    MM32 = zero_arr
    MM33 = xi * (U - 1) / P

    return np.array([[MM00, MM01, MM02, MM03],
                        [MM10, MM11, MM12, MM13],
                        [MM20, MM21, MM22, MM23],
                        [MM30, MM31, MM32, MM33]])


def NNq(U, C, xi, args):
    omega, delt, gamma = args
    G = G_val(U, C, xi, args)
    P = P_val(U, C, xi, args)
    zero_arr = np.zeros_like(xi)

    return np.array([[-np.ones_like(xi), zero_arr, zero_arr, zero_arr],
                    [zero_arr, - G * xi, zero_arr, zero_arr],
                    [zero_arr, zero_arr, - G * xi, zero_arr],
                    [gamma / G, zero_arr, zero_arr, - 1 / P]])

def NNl(U, C, xi, args):
    omega, delt, gamma = args
    G = G_val(U, C, xi, args)

    zero_arr = np.zeros_like(xi)

    return np.array([[zero_arr, zero_arr, G, zero_arr],
                    [zero_arr, zero_arr, zero_arr, zero_arr],
                    [zero_arr, zero_arr, zero_arr, zero_arr],
                    [zero_arr, zero_arr, zero_arr, zero_arr]])
    

def Y_init(l, q, args):
    omega, delt, gamma = args
    U_init, C_init, xi_init = Y0(gamma)

    dGdxi = dGdxi_val(U_init, C_init, xi_init, args)
    dUdxi = dUdxi_val(U_init, C_init, xi_init, args)
    dPdxi = dPdxi_val(U_init, C_init, xi_init, args)

    dG_boundery = - omega * (gamma + 1)/(gamma - 1) - dGdxi
    dUr_boundery = 2 * q / (gamma + 1) - dUdxi
    dUt_boundery = -2 / (gamma + 1)
    dP_boundery = 2 * (2 * (q + 1) - omega) / (gamma + 1) - dPdxi

    return np.array([dG_boundery, dUr_boundery, dUt_boundery, dP_boundery])



def Y_end(Y, U, C, xi, args, q, l):
    omega, delt, gamma = args


    G = G_val(U, C, xi, args)
    P = P_val(U, C, xi, args)

    NN = NNr(U, C, xi, args) + (q * NNq(U, C, xi, args)) + (l * (l + 1) * NNl(U, C, xi, args))
    vec = NN @ Y

    v1 = vec[0]
    v2 = vec[1]
    v4 = vec[3]

    first = - gamma * P * v1
    second = xi * (U - 1) * G * v2
    third = - P * v4

    return first + second + third



def last_per(U, C, xi, args, q, l):

    NN = NNr(U, C, xi, args) + q * NNq(U, C, xi, args) + l * (l + 1) * NNl(U, C, xi, args)
    MM = MMr(U, C, xi, args)

    Y = Y_init(l, q, args)

    for i in range(len(xi)-1):
        Y_prime = np.linalg.solve(MM[:,:,i], NN[:,:,i] @ Y)
        dxi = xi[i+1] - xi[i]
        Y += Y_prime * dxi

    return Y


if __name__ == "__main__":
    gamma = 5/3
    y0 = Y0(gamma)

    omega = 3.1
    delt = find_delta(omega, gamma)
    print(f"Omega: {omega}, Delta: {delt}")
    args = (omega, delt, gamma)

    """
    t = (np.logspace(-40,0,2000) - 1.0) * 0.609485
    t = t[::-1]
    """
    t = np.linspace(0, -100, 1000)

    y = solveODE(ode_sys_by_t, y0, t, DOPRI8_table, args=args)

    U = y[:, 0]
    C = y[:, 1]
    xi = y[:, 2]
    print(f"Final xi: {xi[-1]}")

    line = np.linspace(0, 1, 100)


    plt.plot(U, C, ".")
    plt.plot(line, 1 - line)
    plt.xlabel("U")
    plt.ylabel("C")
    #plt.xlim(0.6, 1.0)
    #plt.ylim(0.0, 0.6)
    plt.grid()
    plt.title("Phase Space")
    plt.show()


    l = 1
    s_arr = np.linspace(-1.1, 0.1, 400)
    alpha = 1/(1-delt)
    q_arr = s_arr/alpha
    Y_arr = np.zeros((4, 400))
    for i, q in enumerate(q_arr):
        Y_arr[:, i] = last_per(U, C, xi, args, q, l)

    Y_arr = np.log(np.abs(Y_arr))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for k, ax in enumerate(axes.flatten()):
        c = ax.plot(s_arr, Y_arr[k, :], ".")

        ax.set_xlabel('s')
        ax.set_ylabel('per')
        print(f"Y_arr[:, {k}]: {Y_arr[:, k]}")

    plt.tight_layout()
    plt.show()

    l_arr = np.logspace(-2, 1, 200)
    s_arr = np.linspace(-1.1, 0.1, 200)
    
    q_arr = s_arr/alpha

    Y_final = np.zeros((len(q_arr), len(l_arr), 4))

    Y_end_arr = np.zeros((len(q_arr), len(l_arr)))

    for i, l in enumerate(l_arr):
        print(f"Processing l={l:.5f}   ", end="\r", flush=True)
        for j, q in enumerate(q_arr):
            Y_final[j, i] = last_per(U, C, xi, args, q, l)
            Y_end_arr[j, i] = np.log(np.abs(Y_end(Y_final[j, i], U[-1], C[-1], xi[-1], args, q, l)))


    Y_final = np.log(np.abs(Y_final))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    titles = ['Y_final[0] (dG)', 'Y_final[1] (dUr)', 'Y_final[2] (dUt)', 'Y_final[3] (dP)', 'Y_end']
    
    # Plot Y_final components
    for k in range(4):
        ax = axes[k // 3, k % 3]
        c = ax.pcolormesh(l_arr, q_arr, Y_final[:, :, k], shading='auto', cmap='viridis')
        fig.colorbar(c, ax=ax)
        ax.set_title(titles[k])
        ax.set_xlabel('l')
        ax.set_ylabel('q')
        ax.set_xscale('log')

    # Plot Y_end
    ax = axes[1, 1]
    c_end = ax.pcolormesh(l_arr, q_arr, Y_end_arr, shading='auto', cmap='viridis')
    fig.colorbar(c_end, ax=ax)
    ax.set_title(titles[4])
    ax.set_xlabel('l')
    ax.set_ylabel('q')
    ax.set_xscale('log')

    # Hide the unused subplot
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    titles = ['Y_final[0] (dG)', 'Y_final[1] (dUr)', 'Y_final[2] (dUt)', 'Y_final[3] (dP)']

    for k, ax in enumerate(axes.flatten()):
        c = ax.pcolormesh(l_arr, q_arr, Y_final[:, :, k], shading='auto', cmap='viridis')
        fig.colorbar(c, ax=ax)
        ax.set_title(titles[k])
        ax.set_xlabel('l')
        ax.set_ylabel('q')
        ax.set_xscale('log')

    plt.tight_layout()
    plt.show()
