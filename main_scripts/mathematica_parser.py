from sympy.parsing.mathematica import parse_mathematica


mat = [
    [
        "(5 theta1 xi - 10 C^2 theta1 xi + 5 C^4 theta1 xi - 5 U^5 ((2 + theta1) xi - 2 xi0) - 5 q (C^2 - (-1 + U)^2)^2 (xi - xi0) + 5 xi omega - 10 C^2 xi omega + 2 C^4 xi omega - 5 xi0 omega + 10 C^2 xi0 omega - 2 C^4 xi0 omega + 10 U^3 (xi (-6 - 5 theta1 + C^2 (2 + theta1) - 2 omega) + 2 xi0 (3 - C^2 + omega)) + 5 U^4 (-xi0 (8 + omega) + xi (8 + 5 theta1 + omega)) - 10 U^2 (xi0 (4 + 3 omega - C^2 (4 + omega)) + xi (-4 - 5 theta1 - 3 omega + C^2 (4 + 3 theta1 + omega))) - 5 U (2 xi0 (-1 - 2 omega + 2 C^2 (1 + omega)) + xi (2 + 5 theta1 + C^4 theta1 + 4 omega - 2 C^2 (2 + 3 theta1 + 2 omega))))/(5 (C^2 - (-1 + U)^2)^2 (-1 + U) (xi - xi0))",
        " -1/(C^2 - (-1 + U)^2)^22^(5 - (5 omega)/3) 3^(3 - omega) 5^(1/2 (-5 + omega))C^4 (1 - U)^(-2 + (4 omega)/(3 (-3 + omega))) xi^((4 (-3 + 2 omega))/(-3 + omega)) (C^2 (1 - U)^((2 omega)/(3 (-3 + omega))) xi^((-6 + 4 omega)/(-3 + omega)))^(1/2 (-1 - omega)) (xi - xi0)^(-theta1 + theta2) (C^2 (-1 + U)^2 (10 + 5 q - 10 U - 4 omega) - 5 (-1 + U)^4 (2 + q - omega) + 2 C^4 omega)",
        " 1/(C^2 - (-1 + U)^2)2^(5 - (5 omega)/3) 3^(3 - omega) 5^(1/2 (-3 + omega))C^4 l (1 + l) (1 - U)^(1 + (4 omega)/(3 (-3 + omega))) xi^((4 (-3 + 2 omega))/(-3 + omega)) (C^2 (1 - U)^((2 omega)/(3 (-3 + omega))) xi^((-6 + 4 omega)/(-3 + omega)))^(1/2 (-1 - omega)) (xi - xi0)^(-theta1 + theta3)",
        " ((xi - xi0)^(-theta1 + theta4) (3 q (C^2 - (-1 + U)^2) - (-1 + U)^2 (10 U - 3 omega)))/(3 (C^2 - (-1 + U)^2)^2 (-1 + U) xi^2)"
    ], 
    [
        "(2^(-5 + (5 omega)/3) 3^(-3 + omega) 5^(1/2 - omega/2) (1 - U)^(2 + (2 omega)/(9 - 3 omega)) xi^((6 - 4 omega)/(-3 + omega)) (C^2 (1 - U)^((2 omega)/(3 (-3 + omega))) xi^((-6 + 4 omega)/(-3 + omega)))^(1/2 (-1 + omega)) (xi - xi0)^(theta1 -theta2) (-10 U + 3 omega))/(C^2 - (-1 + U)^2)^2",
        " (xi0 (15 C^4 + 5 (-1 + U)^3 (-1 + q + U) - C^2 (-1 + U) (-20 + 5 q + 10 U + 3 omega)) + xi (-5 C^4 (3 + theta2) - 5 (-1 + U)^3 (q + (-1 + U) (1 + theta2)) + C^2 (-1 + U) (-20 + 5 q - 10 theta2 + 10 U (1 + theta2) + 3 omega)))/(5 (C^2 - (-1 + U)^2)^2 (xi - xi0))",
        " (C^2 l (1 + l) (xi - xi0)^(-theta2 + theta3))/(C^2 - (-1 + U)^2)",
        " 1/((C^2 - (-1 + U)^2)^2 xi^2)2^(-5 + (5 omega)/3) 3^(-4 + omega) 5^(3/2 - omega/2) (C^2 (1 - U)^((2 omega)/(3 (-3 + omega))) xi^((-6 + 4 omega)/(-3 + omega)))^(1/2 (-3 + omega)) (xi - xi0)^(-theta2 + \theta4) (-3 q (C^2 - (-1 + U)^2) + (-1 + U)^2 (10 U - 3 omega))"
    ], 
    [
        "0",
        " 0",
        " -((-1 + q + 2 U)/(-1 + U)) - (theta3 xi)/(xi - xi0)",
        " -((2^(-5 + (5 omega)/3) 3^(-3 + omega) 5^(3/2 - omega/2) (C^2 (1 - U)^((2 omega)/(3 (-3 + omega))) xi^((-6 + 4 omega)/(-3 + omega)))^(1/2 (-3 + omega)) (xi - xi0)^(-theta3 + \theta4))/((-1 + U) xi^2))"
    ], 
    [
        "(C^4 (-1 + U) xi^2 (xi - xi0)^(theta1 - theta4) (10 U - 3 omega))/(5 (C^2 - (-1 + U)^2)^2)",
        " 1/(C^2 - (-1 + U)^2)^22^(5 - (5 omega)/3) 3^(3 - omega) 5^(1/2 (-5 + omega)) C^6 (1 - U)^((4 omega)/(3 (-3 + omega))) xi^((2 (-9 + 5 omega))/(-3 + omega)) (C^2 (1 - U)^((2 omega)/(3 (-3 + omega))) xi^((-6 + 4 omega)/(-3 + omega)))^(1/2 (-1 - omega)) (xi - xi0)^(theta2 - \theta4) (5 (2 + q - 2 U) (-1 + U)^2 - C^2 (10 + 5 q - 20 U + 3 omega))",
        " 1/(C^2 - (-1 + U)^2)2^(5 - (5 omega)/3) 3^(3 - omega) 5^(1/2 (-3 + omega))C^6 l (1 + l) (1 - U)^(1 + (4 omega)/(3 (-3 + omega))) xi^((2 (-9 + 5 omega))/(-3 + omega)) (C^2 (1 - U)^((2 omega)/(3 (-3 + omega))) xi^((-6 + 4 omega)/(-3 + omega)))^(1/2 (-1 - omega)) (xi - xi0)^(theta3 - theta4)",
        " \(3 q (C^2 - (-1 + U)^2) (-1 + U) (xi - xi0) + U^4 (-((10 + 3 theta4) xi) + 10 xi0) - 3 ((-1 + C^2)^2 theta4 xi + (xi - xi0) omega) + 3 U^3 (-xi0 (10 + omega) + xi (10 + 4 theta4 + omega)) + 3 U^2 (xi (-10 + 2 (-3 + C^2) theta4 - 3 omega) + xi0 (10 + 3 omega)) + U (-xi0 (10 + 9 omega) + xi (10 - 12 (-1 + C^2) theta4 + 9 omega)))/(3 (C^2 - (-1 + U)^2)^2 (xi - xi0))"
    ]
]

expr_mat = [[None for _ in range(len(row))] for row in mat]

for i in range(len(mat)):
    for j in range(len(mat[i])):
        expr_mat[i][j] = parse_mathematica(mat[i][j])
        print(f"def HH{i}{j}(U, C, xi, xi0, omega, thetas, q, l):")
        print("    theta1, theta2, theta3, theta4 = thetas")
        print(f"    return {expr_mat[i][j]}")
        print()
        print()



