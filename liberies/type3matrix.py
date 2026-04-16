import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


@jax.jit
def HH00(sol, xi, q, l):
    U, C = sol.U(xi), sol.C(xi)
    xi0 = sol.xi_final
    omega = sol.omega
    theta1, theta2, theta3, theta4 = sol.thetas
    return (2*C**4*omega*xi - 2*C**4*omega*xi0 + 5*C**4*theta1*xi - 10*C**2*omega*xi + 10*C**2*omega*xi0 - 10*C**2*theta1*xi - 5*U**5*(xi*(theta1 + 2) - 2*xi0) + 5*U**4*(xi*(omega + 5*theta1 + 8) - xi0*(omega + 8)) + 10*U**3*(xi*(C**2*(theta1 + 2) - 2*omega - 5*theta1 - 6) + 2*xi0*(-C**2 + omega + 3)) - 10*U**2*(xi*(C**2*(omega + 3*theta1 + 4) - 3*omega - 5*theta1 - 4) + xi0*(-C**2*(omega + 4) + 3*omega + 4)) - 5*U*(xi*(C**4*theta1 - 2*C**2*(2*omega + 3*theta1 + 2) + 4*omega + 5*theta1 + 2) + 2*xi0*(2*C**2*(omega + 1) - 2*omega - 1)) + 5*omega*xi - 5*omega*xi0 - 5*q*(C**2 - (U - 1)**2)**2*(xi - xi0) + 5*theta1*xi)/(5*(C**2 - (U - 1)**2)**2*(U - 1)*(xi - xi0))


@jax.jit
def HH01(sol, xi, q, l):
    U, C = sol.U(xi), sol.C(xi)
    xi0 = sol.xi_final
    omega = sol.omega
    theta1, theta2, theta3, theta4 = sol.thetas
    return -3**(3 - omega)*5**(omega/2 - 5/2)*C**4*xi**((8*omega - 12)/(omega - 3))*(C**2*xi**((4*omega - 6)/(omega - 3))*(1 - U)**(2*omega/(3*omega - 9)))**(-omega/2 - 1/2)*(1 - U)**(4*omega/(3*omega - 9) - 2)*(xi - xi0)**(-theta1 + theta2)*(2*C**4*omega + C**2*(U - 1)**2*(-10*U - 4*omega + 5*q + 10) - 5*(U - 1)**4*(-omega + q + 2))/(C**2 - (U - 1)**2)**(22**(5 - 5*omega/3))


@jax.jit
def HH02(sol, xi, q, l):
    U, C = sol.U(xi), sol.C(xi)
    xi0 = sol.xi_final
    omega = sol.omega
    theta1, theta2, theta3, theta4 = sol.thetas
    return 2**(5 - 5*omega/3)*3**(3 - omega)*5**(omega/2 - 3/2)*C**4*l*xi**((8*omega - 12)/(omega - 3))*(C**2*xi**((4*omega - 6)/(omega - 3))*(1 - U)**(2*omega/(3*omega - 9)))**(-omega/2 - 1/2)*(1 - U)**(4*omega/(3*omega - 9) + 1)*(l + 1)*(xi - xi0)**(-theta1 + theta3)/(C**2 - (U - 1)**2)


@jax.jit
def HH03(sol, xi, q, l):
    U, C = sol.U(xi), sol.C(xi)
    xi0 = sol.xi_final
    omega = sol.omega
    theta1, theta2, theta3, theta4 = sol.thetas
    return (xi - xi0)**(-theta1 + theta4)*(3*q*(C**2 - (U - 1)**2) - (U - 1)**2*(10*U - 3*omega))/(3*xi**2*(C**2 - (U - 1)**2)**2*(U - 1))


@jax.jit
def HH10(sol, xi, q, l):
    U, C = sol.U(xi), sol.C(xi)
    xi0 = sol.xi_final
    omega = sol.omega
    theta1, theta2, theta3, theta4 = sol.thetas
    return 2**(5*omega/3 - 5)*3**(omega - 3)*5**(1/2 - omega/2)*xi**((6 - 4*omega)/(omega - 3))*(C**2*xi**((4*omega - 6)/(omega - 3))*(1 - U)**(2*omega/(3*omega - 9)))**(omega/2 - 1/2)*(1 - U)**(2*omega/(9 - 3*omega) + 2)*(-10*U + 3*omega)*(xi - xi0)**(theta1 - theta2)/(C**2 - (U - 1)**2)**2


@jax.jit
def HH11(sol, xi, q, l):
    U, C = sol.U(xi), sol.C(xi)
    xi0 = sol.xi_final
    omega = sol.omega
    theta1, theta2, theta3, theta4 = sol.thetas
    return (xi*(-5*C**4*(theta2 + 3) + C**2*(U - 1)*(10*U*(theta2 + 1) + 3*omega + 5*q - 10*theta2 - 20) - 5*(U - 1)**3*(q + (U - 1)*(theta2 + 1))) + xi0*(15*C**4 - C**2*(U - 1)*(10*U + 3*omega + 5*q - 20) + 5*(U - 1)**3*(U + q - 1)))/(5*(C**2 - (U - 1)**2)**2*(xi - xi0))


@jax.jit
def HH12(sol, xi, q, l):
    U, C = sol.U(xi), sol.C(xi)
    xi0 = sol.xi_final
    omega = sol.omega
    theta1, theta2, theta3, theta4 = sol.thetas
    return C**2*l*(l + 1)*(xi - xi0)**(-theta2 + theta3)/(C**2 - (U - 1)**2)


@jax.jit
def HH13(sol, xi, q, l):
    U, C = sol.U(xi), sol.C(xi)
    xi0 = sol.xi_final
    omega = sol.omega
    theta1, theta2, theta3, theta4 = sol.thetas
    return 2**(5*omega/3 - 5)*3**(omega - 4)*5**(3/2 - omega/2)*(C**2*xi**((4*omega - 6)/(omega - 3))*(1 - U)**(2*omega/(3*omega - 9)))**(omega/2 - 3/2)*(xi - xi0)**(theta4 - theta2)*(-3*q*(C**2 - (U - 1)**2) + (U - 1)**2*(10*U - 3*omega))/(xi**2*(C**2 - (U - 1)**2)**2)


@jax.jit
def HH20(sol, xi, q, l):
    U, C = sol.U(xi), sol.C(xi)
    xi0 = sol.xi_final
    omega = sol.omega
    theta1, theta2, theta3, theta4 = sol.thetas
    return jnp.zeros_like(xi)


@jax.jit
def HH21(sol, xi, q, l):
    U, C = sol.U(xi), sol.C(xi)
    xi0 = sol.xi_final
    omega = sol.omega
    theta1, theta2, theta3, theta4 = sol.thetas
    return jnp.zeros_like(xi)


@jax.jit
def HH22(sol, xi, q, l):
    U, C = sol.U(xi), sol.C(xi)
    xi0 = sol.xi_final
    omega = sol.omega
    theta1, theta2, theta3, theta4 = sol.thetas
    return -theta3*xi/(xi - xi0) - (2*U + q - 1)/(U - 1)


@jax.jit
def HH23(sol, xi, q, l):
    U, C = sol.U(xi), sol.C(xi)
    xi0 = sol.xi_final
    omega = sol.omega
    theta1, theta2, theta3, theta4 = sol.thetas
    return -2**(5*omega/3 - 5)*3**(omega - 3)*5**(3/2 - omega/2)*(C**2*xi**((4*omega - 6)/(omega - 3))*(1 - U)**(2*omega/(3*omega - 9)))**(omega/2 - 3/2)*(xi - xi0)**(theta4 - theta3)/(xi**2*(U - 1))


@jax.jit
def HH30(sol, xi, q, l):
    U, C = sol.U(xi), sol.C(xi)
    xi0 = sol.xi_final
    omega = sol.omega
    theta1, theta2, theta3, theta4 = sol.thetas
    return C**4*xi**2*(U - 1)*(10*U - 3*omega)*(xi - xi0)**(theta1 - theta4)/(5*(C**2 - (U - 1)**2)**2)


@jax.jit
def HH31(sol, xi, q, l):
    U, C = sol.U(xi), sol.C(xi)
    xi0 = sol.xi_final
    omega = sol.omega
    theta1, theta2, theta3, theta4 = sol.thetas
    return 3**(3 - omega)*5**(omega/2 - 5/2)*C**6*xi**((10*omega - 18)/(omega - 3))*(C**2*xi**((4*omega - 6)/(omega - 3))*(1 - U)**(2*omega/(3*omega - 9)))**(-omega/2 - 1/2)*(1 - U)**(4*omega/(3*omega - 9))*(xi - xi0)**(-theta4 + theta2)*(-C**2*(-20*U + 3*omega + 5*q + 10) + 5*(U - 1)**2*(-2*U + q + 2))/(C**2 - (U - 1)**2)**(22**(5 - 5*omega/3))


@jax.jit
def HH32(sol, xi, q, l):
    U, C = sol.U(xi), sol.C(xi)
    xi0 = sol.xi_final
    omega = sol.omega
    theta1, theta2, theta3, theta4 = sol.thetas
    return 2**(5 - 5*omega/3)*3**(3 - omega)*5**(omega/2 - 3/2)*C**6*l*xi**((10*omega - 18)/(omega - 3))*(C**2*xi**((4*omega - 6)/(omega - 3))*(1 - U)**(2*omega/(3*omega - 9)))**(-omega/2 - 1/2)*(1 - U)**(4*omega/(3*omega - 9) + 1)*(l + 1)*(xi - xi0)**(theta3 - theta4)/(C**2 - (U - 1)**2)


@jax.jit
def HH33(sol, xi, q, l):
    U, C = sol.U(xi), sol.C(xi)
    xi0 = sol.xi_final
    omega = sol.omega
    theta1, theta2, theta3, theta4 = sol.thetas
    return (U**4*(-xi*(3*theta4 + 10) + 10*xi0) + 3*U**3*(xi*(omega + 4*theta4 + 10) - xi0*(omega + 10)) + 3*U**2*(xi*(-3*omega + 2*theta4*(C**2 - 3) - 10) + xi0*(3*omega + 10)) + U*(xi*(9*omega - 12*theta4*(C**2 - 1) + 10) - xi0*(9*omega + 10)) - 3*omega*(xi - xi0) + 3*q*(C**2 - (U - 1)**2)*(U - 1)*(xi - xi0) - 3*theta4*xi*(C**2 - 1)**2)/(3*(C**2 - (U - 1)**2)**2*(xi - xi0))