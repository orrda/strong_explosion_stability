import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from liberies.solution import solution, solve_PDE

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

def get_residual_fn(omega, gamma):
    def fn(delt):
        # Ensure delt is scalar
        delt = jnp.squeeze(delt)
        HH = (omega - 2*delt)/gamma
        sing_U = (delt + 2 + HH  - jnp.sqrt((delt + 2 + HH)**2 - 8 * HH)) / 4
        
        num_sol = solve_PDE(omega, delt, gamma=gamma, stop_at_sonic=True)
        last_U = num_sol.ys[-1, 0]
        return sing_U - last_U
    return fn

omega = 4.25
gamma = 5/3
fn = get_residual_fn(omega, gamma)

print("Fine sweep around 0.25...")
# 0.25 is the root we found
deltas = jnp.linspace(0.24, 0.26, 21)
for d in deltas:
    try:
        res = fn(d)
        print(f"delta={d:.4f}, residual={res:.4e}")
    except Exception as e:
        print(f"delta={d:.4f}, error={e}")
