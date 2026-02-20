import diffrax
import jax.numpy as jnp
import jax

# Configure JAX to use CPU to avoid any Metal-related noise
jax.config.update("jax_platform_name", "cpu")

def vector_field(t, y, args):
    return -y

term = diffrax.ODETerm(vector_field)
solver = diffrax.Dopri5()
t0 = 0
t1 = 1
dt0 = 0.1
y0 = jnp.array([1.0])
# Ensure we have dense output for evaluation
saveat = diffrax.SaveAt(dense=True) 

print("Solving ODE...")
sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, saveat=saveat)
print("ODE solved.")

t_eval = jnp.array([0.2, 0.5, 0.8])
print(f"Attempting to evaluate at array t: {t_eval}")

try:
    result = sol.evaluate(t_eval)
    print("Success!")
    print("Result shape:", result.shape)
    print("Result:", result)
except Exception as e:
    print("Failed!")
    print(f"Error type: {type(e)}")
    print(f"Error message: {e}")
