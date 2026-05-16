## Quickstart

Implicit pressure-based finite-volume solver for 2D incompressible turbulent flow. For full solver methodology, see [here](#methodology).

1. Install prerequisites (macOS):

```
brew install uv ffmpeg
```

2. Replace `body.dat` with coordinates of an arbitrary closed polygon.

3. Run solver:

```
rm -rf results/ && uv run python main.py
```

4. Animate results:

```
uv run python animate.py && open results/animation.mp4
```

<img width="480" height="480" alt="demo" src="https://github.com/user-attachments/assets/7087071d-97cc-4926-8023-bf938b893879" />

## Configuration

Main files:

| File               | Description                                |
| ------------------ | ------------------------------------------ |
| `main.py`          | PISO algorithm                             |
| `mesh.py`          | Mesh generation and geometry               |
| `reconstruct.py`   | Face and gradient reconstruction           |
| `solver.py`        | Matrix assembly for transport and pressure |
| `turbulence.py`    | Spalart-Allmaras closure and wall function |
| `animate.py`       | Snapshot rendering and animation           |
| `body.dat`         | Body geometry coordinates                  |
| `verification.dat` | Reference pressure coefficient data        |

Solver configuration (in `main.py`):

| Parameter         | Default             | Description                                                              |
| ----------------- | ------------------- | ------------------------------------------------------------------------ |
| `u_inf`           | `1.0`               | Free-stream $x$-velocity                                                 |
| `v_inf`           | `0.0`               | Free-stream $y$-velocity                                                 |
| `Re`              | `1e6`               | Reynolds number                                                          |
| `dt`              | `1e-3`              | Time step size                                                           |
| `nt`              | `1e4`               | Number of time steps                                                     |
| `n_piso`          | `2`                 | Number of PISO inner iterations per time step                            |
| `n_log`           | `1`                 | Log divergence every $n_\text{log}$ steps                                |
| `n_write`         | `50`                | Write snapshot every $n_\text{write}$ steps                              |
| `body_file`       | `"body.dat"`        | Body geometry file (closed polygon, one `x y` pair per line)             |
| `mesh_type`       | `"c_mesh"`          | Mesh topology: `o_mesh` (tri-cells), `c_mesh` (quad-cells)               |
| `check_mesh`      | `False`             | Show interactive mesh viewer before solving                              |
| `gradient_type`   | `"green_gauss"`     | Cell gradient method: `green_gauss`, `least_squares`                     |
| `diffusion_type`  | `"over_relaxed"`    | Diffusion correction: `over_relaxed`, `minimum`, `orthogonal`            |
| `convection_type` | `"barth_jespersen"` | Convection limiter: `barth_jespersen`, `venkatakrishnan`                 |
| `turbulence`      | `True`              | Enable Spalart-Allmaras one-equation RANS model                          |
| `wall_treatment`  | `"wall_function"`   | Wall treatment: `integrate_to_wall`, `wall_function`                     |
| `chi_inf`         | `3.0`               | Free-stream $\chi_\infty = \tilde\nu_\infty/\nu$                         |
| `n_refactor`      | `25`                | Refactorise cached LU preconditioner every $n_\text{refactor}$ steps     |
| `solver_rtol`     | `1e-8`              | Relative tolerance for BiCGStab (momentum, turbulence) and CG (pressure) |
| `solver_maxiter`  | `200`               | Maximum BiCGStab/CG iterations per solve                                 |

Mesh configuration (in `mesh.py`):

| Parameter      | Default | Description                                                |
| -------------- | ------- | ---------------------------------------------------------- |
| `Y_1`          | `1e-3`  | First-cell wall-normal spacing (boundary-layer resolution) |
| `R_FAR`        | `30.0`  | O-mesh: farfield radius                                    |
| `H_FAR`        | `2.0`   | O-mesh: maximum cell size at farfield                      |
| `R_GROWTH`     | `1.2`   | C-mesh: wall-normal cell growth ratio (boundary layer)     |
| `R_OUT`        | `30.0`  | C-mesh: inlet/outlet distance from body                    |
| `LE_LENGTH`    | `0.05`  | C-mesh: leading-edge arc length                            |
| `TE_THICKNESS` | `0.05`  | C-mesh: trailing-edge thickness for streamwise grading     |
| `N_AIRFOIL`    | `100`   | C-mesh: cells along upper/lower body surface               |
| `N_WAKE`       | `100`   | C-mesh: cells along wake                                   |
| `N_LE`         | `25`    | C-mesh: cells along leading-edge arc                       |

## Verification

**Cylinder at $Re = 40$ (laminar, o-mesh)**

<img alt="verification_cylinder" src="https://github.com/user-attachments/assets/69ba81e1-0dec-4aa9-81c0-4a119afb0ec5" />

**Airfoil at $Re = 1 \times 10^6$ (turbulent, wall function, c-mesh)**

<img alt="verification_airfoil" src="https://github.com/user-attachments/assets/e2a87e42-632d-487a-83e0-7e55c863565f" />

## Methodology

<img width="1440" alt="methodology" src="https://github.com/user-attachments/assets/69d06bcc-1fff-4214-9e17-d6201d6fcf78" />
