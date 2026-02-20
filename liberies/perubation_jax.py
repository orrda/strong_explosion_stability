import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp 
import optimistix as optx

from PDE import *
from liberies.solution import solution, solve_PDE

class PerturbationSolver:
    def __init__(self, omega, gamma):
        self.omega = omega
        self.gamma = gamma

        sol = solution(omega=self.omega, gamma=self.gamma)
        self.U = sol.U
        self.C = sol.C
        self.G = sol.G
        self.P = sol.P

    def solve(self):
        # Implement the perturbation solution using JAX
        pass

