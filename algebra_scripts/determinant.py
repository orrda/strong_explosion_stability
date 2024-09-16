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
const = (((gamma + 1)/(gamma - 1))**(1-omega/3)) * ((2 * gamma * (gamma - 1))**((omega - 3)/(3*(gamma - 1)))) 

U = 1 - A * (C ** eta)

G = const * (xi ** (3 - 2*omega)) * (C ** -2)

P = const * (xi ** (5 - 2*omega)) / gamma

Uprime = omega * 3/5 - 3
Pprime = (5 - 2*omega) * const * (xi ** (4 - 2*omega)) / gamma
Gprime = const * (xi ** (3 - 2*omega)) * (C ** -2) * ((3 - 2*omega) * (xi ** -1) - 2 * omega * (C ** - eta)/(5 * A))

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


def inv4(matrix):
    a00 = matrix[0][0]
    a01 = matrix[0][1]
    a02 = matrix[0][2]
    a03 = matrix[0][3]
    a10 = matrix[1][0]
    a11 = matrix[1][1]
    a12 = matrix[1][2]
    a13 = matrix[1][3]
    a20 = matrix[2][0]
    a21 = matrix[2][1]
    a22 = matrix[2][2]
    a23 = matrix[2][3]
    a30 = matrix[3][0]
    a31 = matrix[3][1]
    a32 = matrix[3][2]
    a33 = matrix[3][3]

    det = determinant(matrix)

    b00 = (a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - a11 * a23 * a32 - a12 * a21 * a33 - a13 * a22 * a31) / det
    b01 = (a01 * a23 * a32 + a02 * a21 * a33 + a03 * a22 * a31 - a01 * a22 * a33 - a02 * a23 * a31 - a03 * a21 * a32) / det
    b02 = (a01 * a12 * a33 + a02 * a13 * a31 + a03 * a11 * a32 - a01 * a13 * a32 - a02 * a11 * a33 - a03 * a12 * a31) / det
    b03 = (a01 * a13 * a22 + a02 * a11 * a23 + a03 * a12 * a21 - a01 * a12 * a23 - a02 * a13 * a21 - a03 * a11 * a22) / det
    b10 = (a10 * a23 * a32 + a12 * a20 * a33 + a13 * a22 * a30 - a10 * a22 * a33 - a12 * a23 * a30 - a13 * a20 * a32) / det
    b11 = (a00 * a22 * a33 + a02 * a23 * a30 + a03 * a20 * a32 - a00 * a23 * a32 - a02 * a20 * a33 - a03 * a22 * a30) / det
    b12 = (a00 * a13 * a32 + a02 * a10 * a33 + a03 * a12 * a30 - a00 * a12 * a33 - a02 * a13 * a30 - a03 * a10 * a32) / det
    b13 = (a00 * a12 * a23 + a02 * a13 * a20 + a03 * a10 * a22 - a00 * a13 * a22 - a02 * a10 * a23 - a03 * a12 * a20) / det
    b20 = (a10 * a21 * a33 + a11 * a23 * a30 + a13 * a20 * a31 - a10 * a23 * a31 - a11 * a20 * a33 - a13 * a21 * a30) / det
    b21 = (a00 * a23 * a31 + a01 * a20 * a33 + a03 * a21 * a30 - a00 * a21 * a33 - a01 * a23 * a30 - a03 * a20 * a31) / det
    b22 = (a00 * a11 * a33 + a01 * a13 * a30 + a03 * a10 * a31 - a00 * a13 * a31 - a01 * a10 * a33 - a03 * a11 * a30) / det
    b23 = (a00 * a13 * a21 + a01 * a10 * a23 + a03 * a11 * a20 - a00 * a11 * a23 - a01 * a13 * a20 - a03 * a10 * a21) / det
    b30 = (a10 * a22 * a31 + a11 * a20 * a32 + a12 * a21 * a30 - a10 * a21 * a32 - a11 * a22 * a30 - a12 * a20 * a31) / det
    b31 = (a00 * a21 * a32 + a01 * a22 * a30 + a02 * a20 * a31 - a00 * a22 * a31 - a01 * a20 * a32 - a02 * a21 * a30) / det
    b32 = (a00 * a12 * a31 + a01 * a10 * a32 + a02 * a11 * a30 - a00 * a11 * a32 - a01 * a12 * a30 - a02 * a10 * a31) / det
    b33 = (a00 * a11 * a22 + a01 * a12 * a20 + a02 * a10 * a21 - a00 * a12 * a21 - a01 * a10 * a22 - a02 * a11 * a20) / det

    BB = np.array([
        [b00, b01, b02, b03],
        [b10, b11, b12, b13],
        [b20, b21, b22, b23],
        [b30, b31, b32, b33]
    ])
    return BB


import cmath

def solve_quartic(a, b, c, d, e):
    # Step 1: Normalize the coefficients
    if a == 0:
        raise ValueError("The coefficient 'a' must be non-zero for a quartic equation.")
    
    a3 = b / a
    a2 = c / a
    a1 = d / a
    a0 = e / a

    # Convert to a depressed quartic t^4 + p*t^2 + q*t + r = 0 using x = t - a3/4
    p = a2 - 3 * (a3 ** 2) / 8
    q = a1 - (a3 * a2) / 2 + a3 ** 3 / 8
    r = a0 - (a3 * a1) / 4 + (a2 * a3 ** 2) / 16 - 3 * (a3 ** 4) / 256

    # Solve the resolvent cubic equation y^3 + 2*p*y^2 + (p^2 - 4*r)*y - q^2 = 0
    def solve_cubic(a, b, c, d):
        # Normalize coefficients
        a2 = b / a
        a1 = c / a
        a0 = d / a

        # Convert to depressed cubic t^3 + pt + q = 0 using x = t - a2/3
        p = a1 - a2 ** 2 / 3
        q = 2 * (a2 ** 3) / 27 - (a2 * a1) / 3 + a0

        # Solve the depressed cubic equation
        roots = []
        D0 = p ** 3 / 27
        D1 = q ** 2 / 4
        D = D1 + D0
        
        if D >= 0:
            C = cmath.sqrt(D1 + D0)
            C_plus = cmath.exp(cmath.log(C + q / 2) / 3)
            C_minus = cmath.exp(cmath.log(C - q / 2) / 3)
            root = C_plus + C_minus
            roots.append(root - a2 / 3)
        else:
            theta = cmath.acos(q / (2 * cmath.sqrt(-D0)))
            C = 2 * cmath.sqrt(-p / 3)
            for k in range(3):
                root = C * cmath.cos((theta + 2 * cmath.pi * k) / 3)
                roots.append(root - a2 / 3)

        return roots

    y_roots = solve_cubic(1, 2 * p, p ** 2 - 4 * r, -q ** 2)
    y = max(y_roots, key=abs)

    # Solve the two quadratic equations
    if y == 0:
        u_roots = solve_cubic(1, p, 0, -r)
        v_roots = [0]
    else:
        u_roots = solve_cubic(1, p, 2 * y, y ** 2 - q)
        v_roots = solve_cubic(1, p, -2 * y, y ** 2 + q)
    
    # Calculate the roots of the quartic equation
    roots = []
    for u in u_roots:
        if u >= 0:
            roots.append(cmath.sqrt(u) - a3 / 4)
            roots.append(-cmath.sqrt(u) - a3 / 4)
    for v in v_roots:
        if v >= 0:
            roots.append(cmath.sqrt(v) - a3 / 4)
            roots.append(-cmath.sqrt(v) - a3 / 4)

    return roots






inv_MM = inv4(MM)

matrix = np.dot(inv_MM, NN)

for i in range(4):
    for j in range(4):
        matrix[i][j] = matrix[i][j].subs({A: 1, eps: -omega, delta: 0})
        matrix[i][j] = sp.simplify(matrix[i][j])
        print("simplyfied matrix[{}][{}]".format(i, j))




e = sp.Symbol('e')
matrix = matrix - np.eye(4) * e
det = determinant(matrix)
print("we have a determinant", det)



det = det.as_poly(C)

coeff10 = det.coeff(C, -10)

print("the leading coeff is :",latex(coeff10))





print("the poly is :",latex(det))
with open('/projects/repo/strong_explosion_stability/algebra_scripts/poly.tex', 'w') as file:
    file.write(latex(det))
    print("file written")






print("collected determinant --:", det)



coeff0 = det.coeff(C, 0)
print("the leading coeff is :",latex(coeff0))



"""
sol_q = sp.solve(coeff0, q)
print("the solution for q is :",sol_q)
"""
