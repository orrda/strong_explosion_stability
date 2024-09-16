import sympy as sp
from sympy import latex
import numpy as np
import matplotlib.pyplot as plt





def determinant(matrix):
    m11 = matrix[0][0]
    m12 = matrix[0][1]
    m13 = matrix[0][2]
    m14 = matrix[0][3]
    m21 = matrix[1][0]
    m22 = matrix[1][1]
    m23 = matrix[1][2]
    m24 = matrix[1][3]
    m31 = matrix[2][0]
    m32 = matrix[2][1]
    m33 = matrix[2][2]
    m34 = matrix[2][3]
    m41 = matrix[3][0]
    m42 = matrix[3][1]
    m43 = matrix[3][2]
    m44 = matrix[3][3]

    expression = m11 * m22 * m33 * m44 + m11 * m23 * m34 * m42 + m11 * m24 * m32 * m43 + m12 * m21 * m34 * m43 + m12 * m23 * m31 * m44 + m12 * m24 * m33 * m41 + m13 * m21 * m32 * m44 + m13 * m22 * m34 * m41 + m13 * m24 * m31 * m42 + m14 * m21 * m33 * m42 + m14 * m22 * m31 * m43 + m14 * m23 * m32 * m41 - m11 * m22 * m34 * m43 - m11 * m23 * m32 * m44 - m11 * m24 * m33 * m42 - m12 * m21 * m33 * m44 - m12 * m23 * m34 * m41 - m12 * m24 * m31 * m43 - m13 * m21 * m34 * m42 - m13 * m22 * m31 * m44 - m13 * m24 * m32 * m41 - m14 * m21 * m32 * m43 - m14 * m22 * m33 * m41 - m14 * m23 * m31 * m42

    return expression




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

G = const * (xi ** (3 - 2*omega)) * (C ** -2)

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

NN = np.array([
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

MM = np.array([
    [M11, M12, M13, M14],
    [M21, M22, M23, M24],
    [M31, M32, M33, M34],
    [M41, M42, M43, M44]
])

e = sp.Symbol('e')




matrix = MM - NN * e
#matrix = matrix.subs({A: 1, delta: 0, eps: -omega, gamma: 5/3, omega: 3.1})

for i in range(4):
    for j in range(4):
        matrix[i][j] = matrix[i][j].subs({A: 1, eps: -omega, delta: 0})
        matrix[i][j] = sp.simplify(matrix[i][j])
        print("simplyfied matrix[{}][{}] = {}".format(i, j, latex(matrix[i][j])))




det = determinant(matrix)


poly = sp.expand(det)

poly = sp.collect(poly, C)

coefficients = poly.as_coefficients_dict()
powers = poly.as_powers_dict()

C_powers_coefficients = {}
for power, coefficient in coefficients.items():
    C_powers_coefficients[power] = coefficient

print("C powers and coefficients size:", len(C_powers_coefficients))

print("the poly is :",latex(poly))

"""
poly = sp.simplify(poly)
poly = sp.collect(poly, C)
poly = sp.simplify(poly)

print("the simple poly is :",latex(poly))

with open('/projects/repo/strong_explosion_stability/algebra_scripts/determinant.tex', 'w') as file:
    file.write(latex(poly))
"""
