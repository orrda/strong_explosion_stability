import numpy as np
import matplotlib.pyplot as plt

global gamma, delt, omega, thresh

gamma = 5/3
delt = 0.25
omega = 4.25
thresh = 1e-7

def U_analytic(xi):
    return 2/(gamma + 1)

def C_analytic(xi):
    return xi**3 * np.sqrt(2 * gamma * (gamma - 1)) / (gamma + 1)

def G_analytic(xi):
    return xi**(-8) * (gamma + 1)/(gamma - 1)

def P_analytic(xi):
    return 2/(gamma + 1)


def sonic_xi():
    return ((gamma - 1)/(2 * gamma))**(1/6)


def dGdx_analytic(xi):
    return -8 * xi**(-9) * (gamma + 1)/(gamma - 1)

def dPdx_analytic(xi):
    return 0

def dUdx_analytic(xi):
    return 0


def NNr_analytic(xi):
    U = U_analytic(xi)
    G = G_analytic(xi)
    P = P_analytic(xi)

    dUdxi = dUdx_analytic(xi)
    dGdxi = dGdx_analytic(xi)
    dPdxi = dPdx_analytic(xi)

    NN00 = omega - 3 * U - xi * dUdxi
    NN01 = - xi * dGdxi - 3 * G
    NN02 = 0
    NN03 = 0
    
    NN10 = dPdxi/G
    NN11 = (1 - delt - 2 * U - xi * dUdxi) * G * xi
    NN12 = 0
    NN13 = 0
    
    NN20 = 0
    NN21 = 0
    NN22 = (1 - delt - 2 * U) * G * xi
    NN23 = -1/xi
    
    NN30 = -gamma * (U - 1) * xi * dGdxi/(G**2)
    NN31 = gamma * xi * dGdxi/G
    NN32 = 0
    NN33 = xi * (U - 1) * dPdxi/(P ** 2)
    
    return np.array([[NN00, NN01, NN02, NN03],
                       [NN10, NN11, NN12, NN13],
                       [NN20, NN21, NN22, NN23],
                       [NN30, NN31, NN32, NN33]])


def MM(xi):
    U = U_analytic(xi)
    G = G_analytic(xi)
    P = P_analytic(xi)


    MM00 = xi * (U - 1)
    MM01 = G * xi
    MM02 = 0
    MM03 = 0
    
    MM10 = 0
    MM11 = (U - 1) * G * xi**2
    MM12 = 0
    MM13 = 1
    
    MM20 = 0
    MM21 = 0
    MM22 = (U - 1) * G * xi**2
    MM23 = 0
    
    MM30 = - gamma * (U - 1) * xi / G
    MM31 = 0
    MM32 = 0
    MM33 = xi * (U - 1) / P

    return np.array([[MM00, MM01, MM02, MM03],
                   [MM10, MM11, MM12, MM13],
                   [MM20, MM21, MM22, MM23],
                   [MM30, MM31, MM32, MM33]])


def NNq_analytic(xi):
    G = G_analytic(xi)
    P = P_analytic(xi)

    return np.array([[-1, 0, 0, 0],
                       [0, - G * xi, 0, 0],
                       [0, 0, - G * xi, 0],
                       [gamma / G, 0, 0, - 1 / P]])

def NNl_analytic(xi):
    G = G_analytic(xi)

    return np.array([[0, 0, G, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]])



def boundary_condition(q):
    dGdxi = dGdx_analytic(1)
    dUdxi = dUdx_analytic(1)
    dPdxi = dPdx_analytic(1)

    dG_boundery = - omega * (gamma + 1)/(gamma - 1) - dGdxi
    dUr_boundery = 2 * q / (gamma + 1) - dUdxi
    dUt_boundery = - 2 / (gamma + 1)
    dP_boundery = 2 * (2 * (q + 1) - omega) / (gamma + 1) - dPdxi
    
    return np.array([dG_boundery, dUr_boundery, dUt_boundery, dP_boundery])


def Y_end(q, l, Y):
    xi = sonic_xi()
    NNr = NNr_analytic(xi)
    NNq = NNq_analytic(xi)
    NNl = NNl_analytic(xi)

    P = P_analytic(xi)
    G = G_analytic(xi)
    U = U_analytic(xi)

    NN = NNr + (q * NNq) + (l * (l + 1) * NNl)

    vec = NN @ Y

    v1 = vec[0]
    v2 = vec[1]
    v4 = vec[3]

    first = - gamma * P * v1
    second = xi * (U - 1) * G * v2
    third = - P * v4

    return first + second + third


def Y_sonic(q, l, n):
    last_xi = sonic_xi()
    xi_arr = np.linspace(1,last_xi, n)
    dxi = xi_arr[1] - xi_arr[0]

    Y = boundary_condition(q)

    for xi in xi_arr[:-1]:
        NN = NNr_analytic(xi) + (q * NNq_analytic(xi)) + (l * (l + 1) * NNl_analytic(xi))
        dYdxi = np.linalg.solve(MM(xi), NN @ Y)

        Y = Y + dYdxi * dxi

    return Y


l_arr = np.linspace(0, 2, 50)
q_arr = np.linspace(-1, 0, 50)

Y_arr = np.zeros((len(q_arr), len(l_arr)))

print("Starting computation...")

for i, l in enumerate(l_arr):
    print(f"Processing l={l:.2f}   ", end="\r", flush=True)
    for j, q in enumerate(q_arr):
        Y_S = Y_sonic(q, l, 1000)
        Y_B = Y_end(q, l, Y_S)
        Y_arr[j, i] = Y_B


Q, L = np.meshgrid(q_arr, l_arr, indexing='ij')

plt.figure()
plt.pcolormesh(L, Q, Y_arr, shading='auto', cmap='viridis')
plt.colorbar(label='Difference Norm')
plt.xlabel('l')
plt.ylabel('q')
plt.title('Y_arr as a function of q and l')
plt.show()