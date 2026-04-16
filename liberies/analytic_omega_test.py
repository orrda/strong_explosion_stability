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
    
    NN30 = - gamma * (U - 1) * xi * dGdxi/(G**2)
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

def Y_end(q, l, Y, eps=1e-5):
    # Evaluate at the slightly offset xi where integration stopped
    xi = sonic_xi() + eps
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

def Y_sonic(q, l, eps=1e-5):
    # Integrate from 1 down to slightly before the generic sonic point
    last_xi = sonic_xi() + eps 
    
    Y = boundary_condition(q)
    
    # --- CHANGED SECTION: Replaced solve_ivp with Forward Euler ---
    xi_arr = np.linspace(1.0, last_xi, 1000)
    for i in range(len(xi_arr)-1):
        xi_val = xi_arr[i]
        NN = NNr_analytic(xi_val) + (q * NNq_analytic(xi_val)) + (l * (l + 1) * NNl_analytic(xi_val))
        Y_prime = np.linalg.solve(MM(xi_val), NN @ Y)
        dxi = xi_arr[i+1] - xi_arr[i]
        Y += Y_prime * dxi
    # --------------------------------------------------------------

    return Y


l_arr = np.logspace(-3, 0, 50)
s_arr = np.linspace(-1, 0, 50)
alpha = 1/(1-delt)
q_arr = s_arr

print(f"last xi: {sonic_xi()}")

Y_arr = np.zeros((len(q_arr), len(l_arr), 4))

print("Starting computation...")

eps_val = 1e-4

Y_B_arr = np.zeros((len(q_arr), len(l_arr)))

for i, l in enumerate(l_arr):
    print(f"Processing l={l:.4f}   ", end="\r", flush=True)
    for j, q in enumerate(q_arr):
        Y_S = Y_sonic(q, l, eps=eps_val)
        Y_B = Y_end(q, l, Y_S, eps=eps_val)
        Y_arr[j, i] = Y_S
        Y_B_arr[j, i] = Y_B

print("\nComputation finished.")

Q, L = np.meshgrid(q_arr, l_arr, indexing='ij')

Y_arr = np.log(np.abs(Y_arr))
Y_B_arr = np.log(np.abs(Y_B_arr))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
titles = ['Y_final[0] (dG)', 'Y_final[1] (dUr)', 'Y_final[2] (dUt)', 'Y_final[3] (dP)', 'Y_end']
    
# Plot Y_final components
for k in range(4):
    ax = axes[k // 3, k % 3]
    c = ax.pcolormesh(l_arr, q_arr, Y_arr[:, :, k], shading='auto', cmap='viridis')
    fig.colorbar(c, ax=ax)
    ax.set_title(titles[k])
    ax.set_xlabel('l')
    ax.set_ylabel('q')
    ax.set_xscale('log')

# Plot Y_end
ax = axes[1, 1]
c_end = ax.pcolormesh(l_arr, q_arr, Y_B_arr, shading='auto', cmap='viridis')
fig.colorbar(c_end, ax=ax)
ax.set_title(titles[4])
ax.set_xlabel('l')
ax.set_ylabel('q')
ax.set_xscale('log')

# Hide the unused subplot
axes[1, 2].axis('off')
plt.tight_layout()
plt.show()