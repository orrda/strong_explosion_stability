# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a physics research project analyzing the **stability of strong explosions** (Sedov-Taylor blast waves). The code solves self-similar PDEs for blast wave profiles and computes perturbation eigenvalues to determine stability.

Key parameters:
- `omega`: density power-law exponent (background density ∝ r^{-omega})
- `delta`: similarity exponent (auto-computed if not provided)
- `gamma`: adiabatic index (default 5/3)
- `xi`: self-similar coordinate (1 at shock front, decreasing inward to sonic point)

The `delta` parameter has three regimes:
- `omega ≤ 3`: `delta = (omega - 3) / 2`
- `3 < omega ≤ 3.2554`: `delta = 0`
- `omega > 3.2554`: found by root-finding (U at sonic point matches singularity condition)

## Commands

```bash
# Run tests
pytest

# Run a single test file
pytest tests/test_basic.py

# Run scripts from within the liberies directory (they use relative imports)
cd liberies && python solution.py
cd liberies && python analytic_omega_RK.py

# Run top-level scripts from the project root
python fine_sweep.py
```

JAX requires `jax_enable_x64 = True` and `jax_platform_name = "cpu"` — these are set at the top of most scripts. If running a new script, add:
```python
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
```

## Architecture

### Core ODE Layer (`liberies/PDE.py`, `liberies/PDE_log.py`)

`solve_PDE(omega, delt, ...)` integrates the self-similar blast wave ODE from `xi=1` (shock) inward to the sonic point or `xi=0`. Uses **Diffrax** (Dopri5 solver with PIDController) and returns a dense diffrax solution object supporting `.evaluate(xi)` → `[U, C]`.

- `PDE.py`: integrates in `xi` directly
- `PDE_log.py`: integrates in log-transformed coordinate for improved numerics; `evaluate(t)` returns `[U, C, xi]`

### Solution Layer (`liberies/solution.py`, `liberies/solution_log.py`)

`solution` is an `equinox.Module` wrapping the PDE output. Constructed with `solution(omega, delt=None, gamma=5/3)`. Provides JAX-jittable methods:
- `U(xi)`, `C(xi)` — velocity and sound speed
- `G(xi)`, `P(xi)` — density and pressure (derived from U, C via algebraic relations)
- `dUdx`, `dCdx`, `dGdx`, `dPdx` — spatial derivatives (computed analytically from ODE)

`solution_log` wraps `PDE_log` and takes a log-space parameter `t` instead of `xi`; it additionally exposes `NNr`, `MM`, `NNq`, `NNl` matrices used in the perturbation analysis.

### Perturbation / Stability Analysis

The perturbation state vector `Y = [dG, dUr, dUt, dP]` satisfies:
```
M(xi) dY/dxi = [NNr(xi) + q * NNq(xi) + l(l+1) * NNl(xi)] Y
```
where `q` is the temporal eigenvalue and `l` is the angular harmonic.

Boundary conditions are applied at `xi=1` (shock), integration proceeds to the sonic point. `q` is an eigenvalue when the sonic-point regularity condition is satisfied.

- `liberies/analytic_omega.py`: analytic (constant-profile) stability analysis using `scipy.integrate.solve_ivp`
- `liberies/analytic_omega_RK.py`: same analysis with JAX `odeint`, vectorized over `(q, l)` grids
- `liberies/perubation_class.py`: general perturbation class for numerical (non-analytic) background solutions
- `liberies/perubation_log.py`, `liberies/perubation_magnus.py`: variants using log coordinates or Magnus expansion

### Data and Results

- `DB/`: cached computation results (`.npy` arrays, `.csv`)
- `plots/`: output figures
- `debug_*.py`: standalone debugging scripts for specific numerical issues
- `fine_sweep.py`: parameter sweep over `delta` values near a root

### Branches

- `main`: stable baseline
- `expRK`: current work (RK-based perturbation analysis)
