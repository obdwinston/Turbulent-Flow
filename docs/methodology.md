## Table of Contents

- [Governing Equations](#governing-equations)
  - [Momentum Equation](#momentum-equation)
  - [Continuity Equation](#continuity-equation)
  - [Turbulence Equation](#turbulence-equation)
- [Reconstruction Scheme](#reconstruction-scheme)
  - [Face Reconstruction](#face-reconstruction)
    - [Linear Interpolation](#linear-interpolation)
    - [Non-Orthogonal Correction](#non-orthogonal-correction)
    - [Skewness Correction](#skewness-correction)
  - [Gradient Reconstruction](#gradient-reconstruction)
    - [Green-Gauss Gradient](#green-gauss-gradient)
    - [Least Squares Gradient](#least-squares-gradient)
  - [Reconstruction Algorithm](#reconstruction-algorithm)
- [Boundary Conditions](#boundary-conditions)
- [Transport Equation](#transport-equation)
  - [Diffusion Term](#diffusion-term)
    - [Minimum Correction](#minimum-correction)
    - [Orthogonal Correction](#orthogonal-correction)
    - [Over-Relaxed](#over-relaxed)
    - [Matrix Assembly](#matrix-assembly)
  - [Convection Term](#convection-term)
    - [Barth–Jespersen Limiter](#barthjespersen-limiter)
    - [Venkatakrishnan Limiter](#venkatakrishnan-limiter)
    - [Matrix Assembly](#matrix-assembly-1)
  - [Temporal Term](#temporal-term)
    - [Matrix Assembly](#matrix-assembly-2)
  - [Source Term](#source-term)
    - [Matrix Assembly](#matrix-assembly-3)
- [Pressure Equation](#pressure-equation)
  - [Matrix Assembly](#matrix-assembly-4)
- [Turbulence Model](#turbulence-model)
  - [Spalart-Allmaras Model](#spalart-allmaras-model)
  - [Wall Function](#wall-function)
    - [Spalding's Law](#spaldings-law)
    - [Newton-Raphson Iteration](#newton-raphson-iteration)
  - [Source Assembly](#source-assembly)
- [Solver Algorithm](#solver-algorithm)
- [Appendix](#appendix)

## Governing Equations

### Momentum Equation

$$
\underbrace{\frac{\partial \phi}{\partial t}}_{\text{temporal}} + \underbrace{\nabla \cdot (\phi \mathbf{u})}_{\text{convection}} = \underbrace{\nabla \cdot \left[(\nu + \nu_t) \nabla \phi\right]}_{\text{diffusion}} - \underbrace{\nabla p \cdot \hat{\mathbf{e}}_\phi}_{\text{source}}
$$

$$
\phi \in \{u, v\} \qquad \nu = \frac{1}{\mathrm{Re}} \qquad \nu_\text{eff} = \nu + \nu_t \qquad \hat{\mathbf{e}}_\phi \in \{\hat{\mathbf{i}}, \hat{\mathbf{j}}\}
$$

$\nu_t$ is supplied by the [Spalart-Allmaras model](#spalart-allmaras-model); $\nu_t = 0$ recovers the laminar solver.

### Continuity Equation

$$
\nabla \cdot \mathbf{u} = 0
$$

### Turbulence Equation

$$
\underbrace{\frac{\partial \tilde\nu}{\partial t}}_{\text{temporal}} + \underbrace{\nabla \cdot (\tilde\nu \mathbf{u})}_{\text{convection}} = \underbrace{\nabla \cdot (\frac{\nu + \tilde\nu}{\sigma} \nabla \tilde\nu)}_{\text{diffusion}} + \underbrace{c_{b1} \tilde S \tilde\nu}_{\text{production}} - \underbrace{c_{w1} f_w \left(\frac{\tilde\nu}{d}\right)^{\!2}}_{\text{destruction}} + \underbrace{\frac{c_{b2}}{\sigma} \Vert \nabla \tilde\nu \Vert^2}_{\text{cross-diffusion}}
$$

$$
\tilde\nu \leftarrow \max(\tilde\nu, 0)
$$

$\tilde S$, $f_w$, and $d$ are defined in [Spalart-Allmaras model](#spalart-allmaras-model).

## Reconstruction Scheme

### Face Reconstruction

#### Linear Interpolation

<p align="center"><img src="assets/linear_interpolation.png" alt="Linear interpolation" width="500"></p>

$$
\overline{\phi_f} = \begin{cases} w_f \phi_C + (1 - w_f) \phi_N & \text{interior face} \\ \phi_C & \text{boundary face} \end{cases}
$$

$$
\overline{(\nabla \phi)_f} = \begin{cases} w_f (\nabla \phi)_C + (1 - w_f)(\nabla \phi)_N & \text{interior face} \\ (\nabla \phi)_C & \text{boundary face} \end{cases}
$$

$$
w_f = \frac{\Vert \mathbf{d}_{Nf}\Vert }{\Vert \mathbf{d}_{Cf}\Vert + \Vert \mathbf{d}_{Nf}\Vert }
$$

#### Non-Orthogonal Correction

<p align="center"><img src="assets/non_orthogonal_correction.png" alt="Non-orthogonal correction" width="500"></p>

$$
(\nabla \phi)_f = \overline{(\nabla \phi)_f} + \left[\frac{\phi_N - \phi_C}{\Vert \mathbf{d}_{CN}\Vert } - \overline{(\nabla \phi)_f} \cdot \hat{\mathbf{d}}_{CN}\right] \hat{\mathbf{d}}_{CN}
$$

#### Skewness Correction

<p align="center"><img src="assets/skewness_correction.png" alt="Skewness correction" width="500"></p>

$$
\phi_f = \overline{\phi_f} + (\nabla \phi)_f \cdot \mathbf{d}_{f'f}
$$

### Gradient Reconstruction

#### Green-Gauss Gradient

$$
(\nabla \phi)_C = \frac{1}{V_C} \sum_f \phi_f \mathbf{S}_f
$$

#### Least Squares Gradient

$$
J = \sum_N w_{CN}^2 \left[(\nabla \phi)_C \cdot \mathbf{d}_{CN} - (\phi_N - \phi_C)\right]^2
$$

$$
w_{CN} = \frac{1}{\Vert \mathbf{d}_{CN}\Vert }
$$

$$
\mathbf{M}_C (\nabla \phi)_C = \mathbf{r}_C
$$

$$
\mathbf{M}_C = \sum_N w_{CN}^2 \mathbf{d}_{CN} \mathbf{d}_{CN}^{\top} \qquad \mathbf{r}_C = \sum_N w_{CN}^2 (\phi_N - \phi_C) \mathbf{d}_{CN}
$$

$$
(\nabla \phi)_C = \mathbf{M}_C^{-1} \mathbf{r}_C
$$

### Reconstruction Algorithm

1. Interpolate face scalar $\overline{\phi_f}$.
2. Apply boundary conditions.
3. Compute cell gradient $\overline{(\nabla \phi)_C}$.
4. Interpolate face gradient $\overline{(\nabla \phi)_f}$.
5. Correct non-orthogonality for face gradient $(\nabla \phi)_f$.
6. Correct skewness for face scalar $\phi_f$.
7. Apply boundary conditions.

## Boundary Conditions

| Field         | Body Face                                                      | Farfield Face<br>(Inflow, $\dot{m}_f \le 0$) | Farfield Face<br>(Outflow, $\dot{m}_f > 0$) |
| ------------- | -------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------- |
| $\mathbf{u}$  | $\mathbf{0}$                                                   | $\mathbf{u}_\infty$                          | extrapolated from cell                      |
| $\mathbf{h}$  | $\mathbf{0}$                                                   | $\mathbf{u}_\infty$                          | extrapolated from cell                      |
| $p$           | extrapolated from cell                                         | extrapolated from cell                       | $0$                                         |
| $\dot{m}_f$   | $0$                                                            | $\mathbf{u}_\infty \cdot \mathbf{S}_f$       | from pressure correction                    |
| $\tilde\nu$   | $0$                                                            | $\tilde\nu_\infty$                           | extrapolated from cell                      |
| $\nu + \nu_t$ | $\nu$ (integrate to wall)<br>$\nu_\text{wall}$ (wall function) | extrapolated from cell                       | extrapolated from cell                      |

## Transport Equation

The [momentum](#momentum-equation) and [turbulence](#turbulence-equation) equations share the form

$$
\underbrace{\frac{\partial \phi}{\partial t}}_{\text{temporal}} + \underbrace{\nabla \cdot (\phi \mathbf{u})}_{\text{convection}} - \underbrace{\nabla \cdot (\Gamma \nabla \phi)}_{\text{diffusion}} = \underbrace{S_\phi}_{\text{source}}
$$

with the following specialisations:

| Equation   | $\phi$      | $\Gamma$                   | $S_\phi$                                                                                          |
| ---------- | ----------- | -------------------------- | ------------------------------------------------------------------------------------------------- |
| Momentum   | $u, v$      | $\nu + \nu_t$              | $-\nabla p \cdot \hat{\mathbf{e}}_\phi$                                                           |
| Turbulence | $\tilde\nu$ | $(\nu + \tilde\nu)/\sigma$ | $c_{b1}\tilde S\tilde\nu - c_{w1}f_w(\tilde\nu/d)^2 + (c_{b2}/\sigma)\Vert\nabla\tilde\nu\Vert^2$ |

### Diffusion Term

<p align="center"><img src="assets/diffusion_term.png" alt="Diffusion term" width="500"></p>

$$
\int_V \nabla \cdot (\Gamma \nabla \phi) dV = \sum_f \Gamma_f (\nabla \phi)_f \cdot \mathbf{S}_f
$$

$$
\mathbf{S}_f = \mathbf{E}_f + \mathbf{T}_f
$$

$$
\Gamma_f (\nabla \phi)_f \cdot \mathbf{S}_f = \underbrace{\Gamma_f \Vert \mathbf{E}_f\Vert  \frac{\phi_N - \phi_C}{\Vert \mathbf{d}_{CN}\Vert }}_{\text{implicit}} + \underbrace{\Gamma_f (\nabla \phi)_f \cdot \mathbf{T}_f}_{\text{explicit}}
$$

#### Minimum Correction

$$
\mathbf{E}_f = (\mathbf{S}_f \cdot \hat{\mathbf{d}}_{CN}) \hat{\mathbf{d}}_{CN}
$$

#### Orthogonal Correction

$$
\mathbf{E}_f = \Vert \mathbf{S}_f\Vert  \hat{\mathbf{d}}_{CN}
$$

#### Over-Relaxed

$$
\mathbf{E}_f = \frac{\Vert \mathbf{S}_f\Vert ^2}{\mathbf{S}_f \cdot \hat{\mathbf{d}}_{CN}} \hat{\mathbf{d}}_{CN}
$$

#### Matrix Assembly

<!-- prettier-ignore -->
| Element       | $\mathbf{A}$ Contribution                                                                                                                                                                                                            | $\mathbf{b}$ Contribution                                                                                                                                                    |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Interior Face | $\mathbf{A}[C, C], \mathbf{A}[N, N] \mathrel{+}= +\Gamma_f \Vert \mathbf{E}_f\Vert / \Vert \mathbf{d}_{CN}\Vert $<br>$\mathbf{A}[C, N], \mathbf{A}[N, C] \mathrel{+}= -\Gamma_f \Vert \mathbf{E}_f\Vert / \Vert \mathbf{d}_{CN}\Vert $ | $\mathbf{b}[C] \mathrel{+}= +\Gamma_f (\nabla \phi)_f \cdot \mathbf{T}_f$<br>$\mathbf{b}[N] \mathrel{+}= -\Gamma_f (\nabla \phi)_f \cdot \mathbf{T}_f$ |
| Body Face        | $\mathbf{A}[C, C] \mathrel{+}= +\Gamma_f \Vert \mathbf{E}_f\Vert / \Vert \mathbf{d}_{Cf}\Vert $                                                                                                                                       | $\mathbf{b}[C] \mathrel{+}= +\Gamma_f \Vert \mathbf{E}_f\Vert / \Vert \mathbf{d}_{Cf}\Vert \cdot \phi_f + \Gamma_f (\nabla \phi)_f \cdot \mathbf{T}_f$            |
| Farfield Face (Inflow)  | $\mathbf{A}[C, C] \mathrel{+}= +\Gamma_f \Vert \mathbf{E}_f\Vert / \Vert \mathbf{d}_{Cf}\Vert $                                                                                                                                       | $\mathbf{b}[C] \mathrel{+}= +\Gamma_f \Vert \mathbf{E}_f\Vert / \Vert \mathbf{d}_{Cf}\Vert \cdot \phi_f + \Gamma_f (\nabla \phi)_f \cdot \mathbf{T}_f$            |
| Farfield Face (Outflow) | —                                                                                                                                                                                                                                    | —                                                                                                                                                                            |

### Convection Term

<p align="center"><img src="assets/convection_term.png" alt="Convection term" width="500"></p>

$$
\int_V \nabla \cdot (\phi \mathbf{u}) dV = \sum_f \dot{m}_f \phi_f
$$

$$
\dot{m}_f = \mathbf{u}_f \cdot \mathbf{S}_f
$$

$$
U = \begin{cases} C & \dot{m}_f \ge 0 \\ N & \dot{m}_f < 0 \end{cases}
$$

$$
\dot{m}_f \phi_f = \underbrace{\dot{m}_f \phi_U}_{\text{implicit}} + \underbrace{\dot{m}_f \Psi_U (\nabla \phi)_U \cdot \mathbf{d}_{Uf}}_{\text{explicit}}
$$

$$
\Psi_U \in [0, 1] \qquad \Delta_f = (\nabla \phi)_U \cdot \mathbf{d}_{Uf}
$$

$$
\phi_{\max} = \max(\phi_U, \max_N \phi_N) \qquad \phi_{\min} = \min(\phi_U, \min_N \phi_N)
$$

$$
\Delta_{\max} = \phi_{\max} - \phi_U \qquad \Delta_{\min} = \phi_{\min} - \phi_U
$$

#### Barth–Jespersen Limiter

$$
\Psi_U = \min_f \begin{cases} \min\!\left(1, \dfrac{\Delta_{\max}}{\Delta_f}\right) & \Delta_f > 0 \\[6pt] \min\!\left(1, \dfrac{\Delta_{\min}}{\Delta_f}\right) & \Delta_f < 0 \\[4pt] 1 & \Delta_f = 0 \end{cases}
$$

#### Venkatakrishnan Limiter

$$
\Delta_* = \begin{cases} \Delta_{\max} & \Delta_f > 0 \\ \Delta_{\min} & \Delta_f < 0 \end{cases}
$$

$$
K \in [1, 5]
$$

$$
\epsilon^2 = (K V_U^{1/n_{\dim}})^3
$$

$$
\Psi_U = \min_f \frac{\Delta_*^2 + 2 \Delta_f \Delta_* + \epsilon^2}{\Delta_*^2 + 2 \Delta_f^2 + \Delta_f \Delta_* + \epsilon^2}
$$

#### Matrix Assembly

<!-- prettier-ignore -->
| Element       | $\mathbf{A}$ Contribution                                                                                                                                                                                                | $\mathbf{b}$ Contribution                                                                                                                                                    |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Interior Face | $\mathbf{A}[C, C] \mathrel{+}= +\max(\dot{m}_f, 0)$<br>$\mathbf{A}[C, N] \mathrel{+}= +\min(\dot{m}_f, 0)$<br>$\mathbf{A}[N, N] \mathrel{+}= -\min(\dot{m}_f, 0)$<br>$\mathbf{A}[N, C] \mathrel{+}= -\max(\dot{m}_f, 0)$ | $\mathbf{b}[C] \mathrel{+}= -\dot{m}_f \Psi_U (\nabla \phi)_U \cdot \mathbf{d}_{Uf}$<br>$\mathbf{b}[N] \mathrel{+}= +\dot{m}_f \Psi_U (\nabla \phi)_U \cdot \mathbf{d}_{Uf}$ |
| Body Face        | —                                                                                                                                                                                                                        | —                                                                                                                                                                            |
| Farfield Face (Inflow)  | —                                                                                                                                                                                                                        | $\mathbf{b}[C] \mathrel{+}= -\min(\dot{m}_f, 0) \cdot \phi_f$                                                                                                                |
| Farfield Face (Outflow) | $\mathbf{A}[C, C] \mathrel{+}= +\max(\dot{m}_f, 0)$                                                                                                                                                                      | $\mathbf{b}[C] \mathrel{+}= -\max(\dot{m}_f, 0) \Psi_C (\nabla \phi)_C \cdot \mathbf{d}_{Cf}$                                                                                |

### Temporal Term

$$
\int_V \frac{\partial \phi}{\partial t} dV \approx V_C \frac{\phi_C^{(n+1)} - \phi_C^{(n)}}{\Delta t}
$$

#### Matrix Assembly

<!-- prettier-ignore -->
| Element | $\mathbf{A}$ Contribution                      | $\mathbf{b}$ Contribution                                  |
| ------- | ---------------------------------------------- | ---------------------------------------------------------- |
| Cell    | $\mathbf{A}[C, C] \mathrel{+}= V_C / \Delta t$ | $\mathbf{b}[C] \mathrel{+}= (V_C / \Delta t) \phi_C^{(n)}$ |

### Source Term

$$
\int_V S_\phi dV \approx V_C S_\phi
$$

#### Matrix Assembly

<!-- prettier-ignore -->
| Element | $\mathbf{A}$ Contribution | $\mathbf{b}$ Contribution               |
| ------- | ------------------------- | --------------------------------------- |
| Cell    | —                         | $\mathbf{b}[C] \mathrel{+}= V_C S_\phi$ |

## Pressure Equation

\* see [Appendix](#appendix) for more information

$$
\nabla \cdot (\Gamma \nabla p) = \nabla \cdot \mathbf{h}
$$

$$
\Gamma_C = \frac{V_C}{a_C} \qquad \mathbf{h}_C = \mathbf{u}_C + \frac{(\mathbf{b}_{-p})_C - (\mathbf{A}\mathbf{u})_C}{a_C}
$$

$$
\int_V \nabla \cdot (\Gamma \nabla p) dV = \sum_f \Gamma_f (\nabla p)_f \cdot \mathbf{S}_f
$$

$$
\int_V \nabla \cdot \mathbf{h} dV = \sum_f \mathbf{h}_f \cdot \mathbf{S}_f
$$

$$
\dot{m}_f^* = \begin{cases} \mathbf{h}_f \cdot \mathbf{S}_f + \frac{\Gamma_f}{\Delta t}\left(\dot{m}_f^{(n)} - \mathbf{u}_f^{(n)} \cdot \mathbf{S}_f\right) & \text{interior face} \\ \mathbf{h}_f \cdot \mathbf{S}_f & \text{boundary face} \end{cases}
$$

### Matrix Assembly

<!-- prettier-ignore -->
| Element       | $\mathbf{A}$ Contribution                                                                                                                                                                                                            | $\mathbf{b}$ Contribution                                                                                                                                                                          |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Interior Face | $\mathbf{A}[C, C], \mathbf{A}[N, N] \mathrel{+}= +\Gamma_f \Vert \mathbf{E}_f\Vert / \Vert \mathbf{d}_{CN}\Vert $<br>$\mathbf{A}[C, N], \mathbf{A}[N, C] \mathrel{+}= -\Gamma_f \Vert \mathbf{E}_f\Vert / \Vert \mathbf{d}_{CN}\Vert $ | $\mathbf{b}[C] \mathrel{+}= -\dot{m}_f^* + \Gamma_f (\nabla p)_f \cdot \mathbf{T}_f$<br>$\mathbf{b}[N] \mathrel{+}= +\dot{m}_f^* - \Gamma_f (\nabla p)_f \cdot \mathbf{T}_f$ |
| Body Face        | —                                                                                                                                                                                                                                    | —                                                                                                                                                                                                  |
| Farfield Face (Inflow)  | —                                                                                                                                                                                                                                    | $\mathbf{b}[C] \mathrel{+}= -\mathbf{u}_\infty \cdot \mathbf{S}_f$                                                                                                                                 |
| Farfield Face (Outflow) | $\mathbf{A}[C, C] \mathrel{+}= +\Gamma_f \Vert \mathbf{E}_f\Vert / \Vert \mathbf{d}_{Cf}\Vert $                                                                                                                                       | $\mathbf{b}[C] \mathrel{+}= -\mathbf{h}_f \cdot \mathbf{S}_f + \Gamma_f (\nabla p)_f \cdot \mathbf{T}_f$                                                                                |

## Turbulence Model

### Spalart-Allmaras Model

$$
\chi = \frac{\tilde\nu}{\nu} \qquad f_{v1} = \frac{\chi^3}{\chi^3 + c_{v1}^3} \qquad f_{v2} = 1 - \frac{\chi}{1 + \chi f_{v1}} \qquad \nu_t = \tilde\nu f_{v1}
$$

$$
\Omega = \left\vert \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y} \right\vert \qquad \tilde S = \max\!\left(\Omega + \frac{\tilde\nu f_{v2}}{(\kappa d)^2}, \; c_s \Omega\right) \qquad d \equiv \text{nearest wall distance}
$$

$$
r = \min\!\left(\frac{\tilde\nu}{\tilde S (\kappa d)^2}, \; r_{\max}\right) \qquad g = r + c_{w2}(r^6 - r) \qquad f_w = g \left(\frac{1 + c_{w3}^6}{g^6 + c_{w3}^6}\right)^{\!1/6}
$$

| Constant   | Value                                   |
| ---------- | --------------------------------------- |
| $\sigma$   | $2/3$                                   |
| $\kappa$   | $0.41$                                  |
| $c_{b1}$   | $0.1355$                                |
| $c_{b2}$   | $0.622$                                 |
| $c_{v1}$   | $7.1$                                   |
| $c_{w1}$   | $c_{b1}/\kappa^2 + (1 + c_{b2})/\sigma$ |
| $c_{w2}$   | $0.3$                                   |
| $c_{w3}$   | $2.0$                                   |
| $c_s$      | $0.3$                                   |
| $r_{\max}$ | $10$                                    |

### Wall Function

#### Spalding's Law

(implicit in $u_\tau$)

$$
y^+ = u^+ + \frac{1}{E}\left[\exp(\kappa u^+) - 1 - \kappa u^+ - \frac{(\kappa u^+)^2}{2} - \frac{(\kappa u^+)^3}{6}\right]
$$

$$
y^+ = \frac{u_\tau y_c}{\nu} \qquad u^+ = \frac{u_\parallel}{u_\tau} \qquad E = 9.8
$$

#### Newton-Raphson Iteration

(solve for $u_\tau$)

$$
u_\tau^{(0)} = \sqrt{\frac{\nu u_\parallel}{y_c}} \qquad u_\tau \leftarrow \max(u_\tau - f/f', \; \epsilon)
$$

$$
f(u_\tau) = y^+ - u^+ - \frac{1}{E}\left[\exp(\kappa u^+) - 1 - \kappa u^+ - \frac{(\kappa u^+)^2}{2} - \frac{(\kappa u^+)^3}{6}\right]
$$

$$
f'(u_\tau) = \frac{y_c}{\nu} + \frac{u_\parallel}{u_\tau^2} + \frac{\kappa u^+ \left[\exp(\kappa u^+) - 1 - \kappa u^+ - (\kappa u^+)^2/2\right]}{E\, u_\tau}
$$

$$
\nu_\text{wall} = \frac{u_\tau^2 y_c}{u_\parallel}
$$

### Source Assembly

Temporal, convection, and diffusion follow the [Transport Equation](#transport-equation) assembly with $\phi = \tilde\nu$ and $\Gamma_f = (\nu + \tilde\nu_f)/\sigma$. Production and cross-diffusion are added to $\mathbf{b}$, while destruction is linearised semi-implicitly to put a positive coefficient on the $\mathbf{A}$ diagonal.

<!-- prettier-ignore -->
| Element                | $\mathbf{A}$ Contribution                                                  | $\mathbf{b}$ Contribution                                                            |
| ---------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Cell (Production)      | —                                                                          | $\mathbf{b}[C] \mathrel{+}= +c_{b1} \tilde S_C \tilde\nu_C V_C$                       |
| Cell (Destruction)     | $\mathbf{A}[C, C] \mathrel{+}= +c_{w1} f_{w,C}\, \tilde\nu_C V_C / d_C^2$   | —                                                                                    |
| Cell (Cross-Diffusion) | —                                                                          | $\mathbf{b}[C] \mathrel{+}= +(c_{b2}/\sigma) \Vert (\nabla \tilde\nu)_C \Vert^2 V_C$  |

## Solver Algorithm

Initialise $\mathbf{u}$, $p$, $\dot{m}_f$, and (if turbulence is enabled) $\tilde\nu$, $\nu_t$ from the initial and boundary conditions.

For each time step $n$:

1. Save $\dot{m}_f^{(n)} \leftarrow \dot{m}_f$; reclassify each farfield face as inflow ($\dot{m}_f \le 0$) or outflow ($\dot{m}_f > 0$).

2. **Momentum predictor**:
   1. For $\phi \in \{u, v\}$: reconstruct $\phi_f$ from $\phi$; save $\phi_f^{(n)} \leftarrow \phi_f$.
   2. Assemble $\mathbf{A}$ (shared by $u$, $v$) from temporal, diffusion ($\Gamma_f = \nu + \nu_t$), and convection.
   3. For $\phi \in \{u, v\}$:
      1. Assemble $\mathbf{b}$ from temporal, diffusion, and convection; save $\mathbf{b}_{-p} \leftarrow \mathbf{b}$.
      2. Add source $-\nabla p \cdot \hat{\mathbf{e}}_\phi$ to $\mathbf{b}$; solve $\mathbf{A} \phi = \mathbf{b}$.

3. Compute $\Gamma_C = V_C / \mathbf{A}_{CC}$; reconstruct $\Gamma_f$ from $\Gamma$.

4. Assemble pressure matrix $\mathbf{A}_p$ from $\Gamma_f$.

5. **Pressure corrector** (repeat $n_{\text{piso}}$ times):
   1. Compute $\mathbf{h}_C = \mathbf{u}_C + ((\mathbf{b}_{-p})_C - (\mathbf{A}\mathbf{u})_C) / \mathbf{A}_{CC}$; reconstruct $\mathbf{h}_f$ from $\mathbf{h}$.
   2. Compute $\dot{m}_f^*$ from $\mathbf{h}_f$.
   3. Assemble $\mathbf{b}$; solve $\mathbf{A}_p p = \mathbf{b}$.
   4. Correct $\dot{m}_f \leftarrow \dot{m}_f^* - \Gamma_f (\nabla p)_f \cdot \mathbf{S}_f$; reconstruct $p_f$ from $p$.
   5. Correct $\mathbf{u}_C \leftarrow \mathbf{h}_C - \Gamma_C (\nabla p)_C$.

6. **Turbulence corrector** (if enabled):
   1. Reconstruct $\tilde\nu_f$ from $\tilde\nu$; evaluate closure $\tilde S$, $f_w$ from $\tilde\nu^{(n)}$, $\Omega$, $d$.
   2. Assemble $\mathbf{A}_{\tilde\nu}$ from temporal, diffusion ($\Gamma_f = (\nu + \tilde\nu_f)/\sigma$), convection, and destruction.
   3. Assemble $\mathbf{b}$ from temporal, diffusion, convection, production, and cross-diffusion.
   4. Solve $\mathbf{A}_{\tilde\nu} \tilde\nu = \mathbf{b}$; clamp $\tilde\nu \leftarrow \max(\tilde\nu, 0)$.
   5. Update $\nu_t = \tilde\nu f_{v1}$; refresh $(\nu_\text{eff})_f = \nu + (\nu_t)_f$ with body-face override ($\nu$ or $\nu_\text{wall}$).

## Appendix

### Pressure Equation

Let $\mathbf{u}_C := [u_C, v_C]^\top$ denote the cell-centred velocity. The discrete momentum equation at cell $C$ separates the pressure source from the rest of the right-hand side:

$$
a_C \mathbf{u}_C + \sum_{N \in C} a_N \mathbf{u}_N = \mathbf{b}_C = (\mathbf{b}_{-p})_C - (\nabla p)_C V_C
$$

Solving for $\mathbf{u}_C$ and grouping the pressure-free part as $\mathbf{h}_C$ with prefactor $\Gamma_C := V_C / a_C$:

$$
\mathbf{u}_C = \frac{(\mathbf{b}_{-p})_C}{a_C} - \sum_{N \in C} \frac{a_N}{a_C} \mathbf{u}_N - \frac{V_C}{a_C} (\nabla p)_C
$$

$$
\mathbf{h}_C := \frac{(\mathbf{b}_{-p})_C}{a_C} - \sum_{N \in C} \frac{a_N}{a_C} \mathbf{u}_N \qquad \Gamma_C := \frac{V_C}{a_C}
$$

$$
\mathbf{u}_C = \mathbf{h}_C - \Gamma_C (\nabla p)_C
$$

Note that the off-diagonal sum in $\mathbf{h}_C$ can be rewritten using the matrix-vector product $\mathbf{A}\mathbf{u}$. Since

$$
(\mathbf{A}\mathbf{u})_C = a_C \mathbf{u}_C + \sum_{N \in C} a_N \mathbf{u}_N
$$

we have

$$
\sum_{N \in C} \frac{a_N}{a_C} \mathbf{u}_N = \frac{(\mathbf{A}\mathbf{u})_C}{a_C} - \mathbf{u}_C
$$

Substituting into the definition of $\mathbf{h}_C$:

$$
\mathbf{h}_C = \frac{(\mathbf{b}_{-p})_C}{a_C} - \sum_{N \in C} \frac{a_N}{a_C} \mathbf{u}_N = \mathbf{u}_C + \frac{(\mathbf{b}_{-p})_C - (\mathbf{A}\mathbf{u})_C}{a_C}
$$

The code uses this latter form because $\mathbf{A}\mathbf{u}$ is a single matrix-vector multiplication.

Vectorising over all cells and imposing continuity $\nabla \cdot \mathbf{u} = 0$ yields the pressure Poisson equation:

$$
\mathbf{u} = \mathbf{h} - \Gamma \nabla p
$$

$$
\therefore \nabla \cdot (\Gamma \nabla p) = \nabla \cdot \mathbf{h}
$$

Solving this Poisson equation gives $p$, which projects $\mathbf{u}$ onto the divergence-free constraint via $\mathbf{u} = \mathbf{h} - \Gamma \nabla p$.

### Rhie-Chow Residual

On a **staggered** grid the face mass flux is updated directly via the face-centred relation:

$$
\dot{m}_f \leftarrow \dot{m}_f^* - \Gamma_f (\nabla p)_f \cdot \mathbf{S}_f
$$

On a **collocated** grid the cell velocity is updated and then interpolated to the face:

$$
\mathbf{u}_C \leftarrow \mathbf{h}_C - \Gamma_C (\nabla p)_C
$$

$$
\mathbf{u}_f \leftarrow w_f \mathbf{u}_C + (1 - w_f) \mathbf{u}_N
$$

The face flux from this interpolation differs from the face-centred update:

$$
\dot{m}_f - \mathbf{u}_f \cdot \mathbf{S}_f \neq 0
$$

Rhie-Chow removes this residual by computing $\dot{m}_f$ via a face-centred update directly, recovering staggered-grid behaviour on a collocated mesh.

### Rhie-Chow Correction

For unsteady flow, the predictor RHS without the pressure source contains a temporal contribution:

$$
(\mathbf{b}_{-p})_C = \frac{V_C}{\Delta t} \mathbf{u}_C^{(n)} + \text{convection} + \text{diffusion}
$$

Dividing by $a_C$ produces a corresponding $\Gamma_C / \Delta t$ term in $\mathbf{h}_C$:

$$
\mathbf{h}_C = \frac{(\mathbf{b}_{-p})_C}{a_C} - \sum_{N \in C} \frac{a_N}{a_C} \mathbf{u}_N = \frac{\Gamma_C}{\Delta t} \mathbf{u}_C^{(n)} + \text{rest}
$$

Interpolating to the face and projecting onto $\mathbf{S}_f$:

$$
\mathbf{h}_f = \frac{\Gamma_f}{\Delta t} \mathbf{u}_f^{(n)} + (\text{rest})_f
$$

$$
\mathbf{h}_f \cdot \mathbf{S}_f = \frac{\Gamma_f}{\Delta t}\left(\mathbf{u}_f^{(n)} \cdot \mathbf{S}_f\right) + (\text{rest})_f \cdot \mathbf{S}_f
$$

Replacing the collocated face-velocity contribution $\mathbf{u}_f^{(n)} \cdot \mathbf{S}_f$ with the face-centred mass flux $\dot{m}_f^{(n)}$ from the previous step gives the Rhie-Chow-corrected predictor flux:

$$
\therefore \dot{m}_f^* \leftarrow \mathbf{h}_f \cdot \mathbf{S}_f + \frac{\Gamma_f}{\Delta t}\left(\dot{m}_f^{(n)} - \mathbf{u}_f^{(n)} \cdot \mathbf{S}_f\right)
$$

The correction $\dot{m}_f^{(n)} - \mathbf{u}_f^{(n)} \cdot \mathbf{S}_f$ replaces the collocated-interpolated face velocity contribution with the face-centred (Rhie-Chow-corrected) mass flux from the previous step.
