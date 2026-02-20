
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optimistix as optx
from liberies.solution import solution, solve_PDE

# Enable 64-bit precision for better gradient stability
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

def get_residual_fn(omega, gamma):
    def fn(delt):
        HH = (omega - 2*delt)/gamma
        sing_U = (delt + 2 + HH  - jnp.sqrt((delt + 2 + HH)**2 - 8 * HH)) / 4
        
        # We need to reshape delt to be a scalar if it comes in as a 0-d array
        delt_val = jnp.squeeze(delt)
        
        num_sol = solve_PDE(omega, delt_val, gamma=gamma, stop_at_sonic=True)
        # Check if we actually stopped? 
        # But for now just get the value.
        last_U = num_sol.ys[-1, 0]
        return sing_U - last_U
    return fn

omega = 4.25
gamma = 5/3
solution_inst = solution(omega, gamma=gamma)
fn = get_residual_fn(omega, gamma)

# 1. Sweep values to see the function shape
print("Sweeping delta values...")
deltas = jnp.linspace(0.01, 1.0, 20)
residuals = []
for d in deltas:
    try:
        res = fn(d)
        residuals.append(float(res))
        print(f"delta={d:.4f}, residual={res:.4e}")
    except Exception as e:
        print(f"delta={d:.4f}, error={e}")
        residuals.append(float('nan'))

# 2. Check Gradients
print("\nChecking gradients at specific points...")
grad_fn = jax.grad(fn)
value_and_grad_fn = jax.value_and_grad(fn)

test_points = [0.1, 0.25, 0.5]
for d in test_points:
    try:
        val, grad = value_and_grad_fn(d)
        print(f"delta={d}: val={val:.4e}, grad={grad:.4e}")
    except Exception as e:
        print(f"delta={d}: Error computing grad: {e}")

