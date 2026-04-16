import numpy as np
from solutione import *
from functools import reduce

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


def inverse_MM(MM):
    MM_inv = np.zeros_like(MM)
    for i in range(MM.shape[2]):
        MM_inv[:,:,i] = np.linalg.inv(MM[:,:,i])
    return MM_inv

def S_matrix(U, C, xi, args):
    omega, delt, gamma = args
    last_xi = xi[-1]
    x_arr = xi - last_xi
    phi = (gamma - 1) * omega / (2 * (omega - 3 * gamma))
    dG_exp = 2 * phi
    dU_exp = -1
    dP_exp = 0
    zero_arr = np.zeros_like(x_arr)

    matrix = np.array([[x_arr**dG_exp, zero_arr, zero_arr, zero_arr],
                    [zero_arr, x_arr**dU_exp, zero_arr, zero_arr],
                    [zero_arr, zero_arr, x_arr**dU_exp, zero_arr],
                    [zero_arr, zero_arr, zero_arr, zero_arr]])
    return np.transpose(matrix, (2, 0, 1))

def inverse_S(U, C, xi, args):
    omega, delt, gamma = args
    last_xi = xi[-1]
    x_arr = xi - last_xi
    phi = (gamma - 1) * omega / (2 * (omega - 3 * gamma))
    dG_exp = -2 * phi
    dU_exp = 1
    dP_exp = 0
    zero_arr = np.zeros_like(x_arr)

    return np.array([[x_arr**dG_exp, zero_arr, zero_arr, zero_arr],
                    [zero_arr, x_arr**dU_exp, zero_arr, zero_arr],
                    [zero_arr, zero_arr, x_arr**dU_exp, zero_arr],
                    [zero_arr, zero_arr, zero_arr, x_arr**dP_exp]])

def S_prime(U, C, xi, args):
    omega, delt, gamma = args
    last_xi = xi[-1]
    x_arr = xi - last_xi
    phi = (gamma - 1) * omega / (2 * (omega - 3 * gamma))
    dG_exp = 2 * phi
    dU_exp = -1
    dP_exp = 0
    zero_arr = np.zeros_like(x_arr)

    matrix = np.array([[dG_exp * x_arr**(dG_exp - 1), zero_arr, zero_arr, zero_arr],
                    [zero_arr, dU_exp * x_arr**(dU_exp - 1), zero_arr, zero_arr],
                    [zero_arr, zero_arr, dU_exp * x_arr**(dU_exp - 1), zero_arr],
                    [zero_arr, zero_arr, zero_arr, dP_exp * x_arr**(dP_exp - 1)]])
    return np.transpose(matrix, (2, 0, 1))


def A_matrix(U, C, xi, q, l, args):
    NN = NNr(U, C, xi, args) + q * NNq(U, C, xi, args) + l * (l + 1) * NNl(U, C, xi, args)
    MM_inv = inverse_MM(MMr(U, C, xi, args))
    SS = S_matrix(U, C, xi, args)
    SS_inv = inverse_S(U, C, xi, args)

    return SS_inv @ ( MM_inv @ NN @ SS - S_prime(U, C, xi, args))


def Y_last(U, C, xi, q_arr, l_arr, args):
    Y_last_arr = np.zeros((len(q_arr), len(l_arr), 4))


    MM_inv = inverse_MM(MMr(U, C, xi, args))
    MM_inv = np.transpose(MM_inv, (2, 0, 1))

    Nr = NNr(U, C, xi, args)
    Nq = NNq(U, C, xi, args)
    Nl = NNl(U, C, xi, args)

    xi_diff = np.diff(xi)

    SS = S_matrix(U, C, xi, args)
    SS_inv = inverse_S(U, C, xi, args)
    SS_prime = S_prime(U, C, xi, args)
    SS_inv = np.transpose(SS_inv, (2, 0, 1))

    print("shape of MM_inv:", MM_inv.shape)
    print("shape of Nr:", Nr.shape)
    print("shape of Nq:", Nq.shape)
    print("shape of Nl:", Nl.shape)
    print("shape of SS:", SS.shape)
    print("shape of SS_inv:", SS_inv.shape)
    print("shape of SS_prime:", SS_prime.shape)

    for i, l in enumerate(l_arr):
        print(f"Processing l={l:.5f}   ", end="\r", flush=True)
        for j, q in enumerate(q_arr):
            Y_1 = Y_init(l, q, args)
            NN = Nr + q * Nq + l * (l + 1) * Nl
            NN = np.transpose(NN, (2, 0, 1))
            A = SS_inv @ ( MM_inv @ NN @ SS - SS_prime)
            A = np.eye(4) + xi_diff[:, None, None] * A[:-1,:,:]

            last_A = reduce(np.matmul, A)

            Y_last_arr[i, j] = last_A.dot(Y_1)

    return Y_last_arr





if __name__ == "__main__":
    gamma = 5/3
    y0 = Y0(gamma)

    omega = 3.1
    delt = 0.
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

    l_arr = np.logspace(-2, 1, 100)
    q_arr = np.linspace(-1, 0, 100)

    Y_last_arr = Y_last(U, C, xi, q_arr, l_arr, args)
    
    Y_final = np.log(np.abs(Y_last_arr))
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    titles = ['Y_final[0] (dG)', 'Y_final[1] (dUr)', 'Y_final[2] (dUt)', 'Y_final[3] (dP)', 'Y_end']
    
    # Plot Y_final components
    for k in range(4):
        ax = axes[k // 2, k % 2]
        c = ax.pcolormesh(l_arr, q_arr, Y_final[:, :, k], shading='auto', cmap='viridis')
        fig.colorbar(c, ax=ax)
        ax.set_title(titles[k])
        ax.set_xlabel('l')
        ax.set_ylabel('q')
        ax.set_xscale('log')


    plt.tight_layout()
    plt.show()
