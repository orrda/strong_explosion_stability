import sympy as sp

C = sp.Symbol('C')
omega = sp.Symbol('omega')
gamma = sp.Symbol('gamma')
xi = sp.Symbol('xi')
q = sp.Symbol('q')
l = sp.Symbol('l')
delta = sp.Symbol('delta')
eps = sp.Symbol('eps')
eta = - (2 * gamma)*(3+eps/gamma)/((gamma - 1)*eps)
A = sp.Symbol('A')

lambd = (2*delta + omega*(gamma - 1))/(3 - omega)
const = ((gamma + 1)**(gamma + 1))/(2*gamma*((gamma - 1)**(gamma + 2)))

U = 1 - A * (C ** eta)

G = (const * (xi ** (2 - 3 * lambd)) * (C ** (2 - (eta * lambd)))) ** (1/(gamma - 1 + lambd))

P = (xi ** 2) * (C ** 2) * G/gamma

Uprime = sp.diff(U, xi)
Pprime = sp.diff(P, xi)
Gprime = sp.diff(G, xi)

N11 = omega - q - 3*U - xi * Uprime
N12 = -xi*Gprime - 3 * G
N13 = l * (l + 1) * G
N14 = 0
N21 = Pprime/G
N22 = xi * G * (1 - delta - q - 2*U - xi * Uprime)
N23 = 0
N24 = 0
N31 = 0
N32 = 0
N33 = xi * G * (1 - delta - q - 2*U)
N34 = -1/xi
N41 = (gamma/G) * (q - xi * (U - 1) * Gprime/G)
N42 = xi * gamma * Gprime/G
N43 = 0
N44 = xi * (U - 1) * Pprime/(P**2) - q/P

NN = sp.Matrix([
    [N11, N12, N13, N14],
    [N21, N22, N23, N24],
    [N31, N32, N33, N34],
    [N41, N42, N43, N44]
])

M11 = xi * (U - 1)
M12 = xi * G
M13 = 0
M14 = 0
M21 = 0
M22 = (U - 1) * (xi ** 2) * G
M23 = 0
M24 = 1
M31 = 0
M32 = 0
M33 = (U - 1) * (xi ** 2) * G
M34 = 0
M41 = - gamma * xi * (U - 1)/G
M42 = 0
M43 = 0
M44 = xi * (U - 1)/P

MM = sp.Matrix([
    [M11, M12, M13, M14],
    [M21, M22, M23, M24],
    [M31, M32, M33, M34],
    [M41, M42, M43, M44]
])

e = sp.Symbol('e')

matrix = MM - e * NN
matrix = matrix.subs({A: 1, delta: 0, eps: -omega, gamma: 5/3, omega: 1, l: 2, q: 0.1, xi: 0.5, C:0.00001})
det = matrix.det()
