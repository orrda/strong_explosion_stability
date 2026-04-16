import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp  # Added JAX import
import optimistix as optx
import equinox as eqx

from solution import solution, solve_PDE

@jax.jit
def comm(A, B):
    return A @ B - B @ A


@jax.jit
def inverse(A):
    