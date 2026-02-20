
import jax
import jax.numpy as jnp
import optimistix as optx
from liberies.solution import solve_PDE

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

def fn(delt, args):
    omega, gamma = args
    delt = jnp.squeeze(delt)
    # Print shape to debug if it is batched
    # jax.debug.print("delt shape: {}", delt.shape)
    
    HH = (omega - 2*delt)/gamma
    sing_U = (delt + 2 + HH  - jnp.sqrt((delt + 2 + HH)**2 - 8 * HH)) / 4
    
    num_sol = solve_PDE(omega, delt, gamma=gamma, stop_at_sonic=True)
    
    last_U = num_sol.ys[-1, 0]
    return sing_U - last_U

omega = 4.25
gamma = 5/3

solver = optx.Bisection(rtol=1e-5, atol=1e-5)
# Bracket known to contain solution (0.25)
y0 = jnp.array([0.1, 0.4]) 
args = (omega, gamma)

print("Running Bisection...")
try:
    point = optx.root_find(fn, solver, y0, args=args, throw=False)
    print("Status:", point.result)
    print("Found:", point.value)
    print("Residual:", fn(point.value, args))
except Exception as e:
    print("Crashed:", e)
    import traceback
    traceback.print_exc()
