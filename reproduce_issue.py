
import jax
import jax.numpy as jnp
import optimistix as optx
from liberies.solution import solution, solve_PDE

jax.config.update("jax_platform_name", "cpu")

def fn(delt, args):
    omega, gamma = args
    # Ensure delt is scalar
    delt = jnp.squeeze(delt)
    
    HH = (omega - 2*delt)/gamma
    sing_U = (delt + 2 + HH  - jnp.sqrt((delt + 2 + HH)**2 - 8 * HH)) / 4
    
    num_sol = solve_PDE(omega, delt, gamma=gamma, stop_at_sonic=True)
    
    last_U = num_sol.ys[-1, 0]
    return sing_U - last_U

omega = 4.25
gamma = 5/3
y0_values = [0.2, 0.25, 0.3]

for y0 in y0_values:
    solver = optx.Newton(rtol=1e-5, atol=1e-5)
    args = (omega, gamma)
    print(f"\nTesting Newton with y0={y0}")
    try:
        point = optx.root_find(fn, solver, y0, args=args, throw=False, max_steps=50)
        print("Status:", point.result)
        print("Found:", point.value)
        print("Residual:", fn(point.value, args))
    except Exception as e:
        print("Crashed:", e)
