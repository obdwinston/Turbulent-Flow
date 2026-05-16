import numpy as np

from reconstruct import interpolate_face


SIGMA = 2.0 / 3.0
KAPPA = 0.41
C_B1 = 0.1355
C_B2 = 0.622
C_V1 = 7.1
C_W1 = C_B1 / KAPPA**2 + (1.0 + C_B2) / SIGMA
C_W2 = 0.3
C_W3 = 2.0
C_S = 0.3
R_MAX = 10.0

E = 9.8
KU_PLUS_MAX = 50.0
MAX_ITER = 10
RTOL = 1e-2


def compute_nu_t(nu_t, nu_tilde, nu):
    chi = nu_tilde / nu
    chi3 = chi * chi * chi
    nu_t[:] = nu_tilde * chi3 / (chi3 + C_V1**3)


def compute_closure(S_tilde, f_w, nu_tilde, nu, Omega, d):
    chi = nu_tilde / nu
    chi3 = chi * chi * chi
    f_v1 = chi3 / (chi3 + C_V1**3)
    f_v2 = 1.0 - chi / (1.0 + chi * f_v1)

    kd2 = (KAPPA * d) ** 2
    S_bar = nu_tilde * f_v2 / kd2
    np.maximum(Omega + S_bar, C_S * Omega, out=S_tilde)

    r = np.minimum(nu_tilde / (S_tilde * kd2 + 1e-30), R_MAX)
    g = r + C_W2 * (r**6 - r)
    f_w[:] = g * ((1.0 + C_W3**6) / (g**6 + C_W3**6)) ** (1.0 / 6.0)


def add_production(mesh, b, S_tilde, nu_tilde):
    b += C_B1 * S_tilde * nu_tilde * mesh.cell_V_C


def add_destruction(mesh, A_coo, nu_tilde, f_w):
    n_cells = len(mesh.cells)
    d = mesh.cell_d
    coeff = C_W1 * f_w * nu_tilde * mesh.cell_V_C / (d * d)
    A_coo.data.append(coeff)
    A_coo.rows.append(np.arange(n_cells, dtype=np.int64))
    A_coo.cols.append(np.arange(n_cells, dtype=np.int64))


def add_cross_diffusion(mesh, b, grad_nu_tilde):
    b += (C_B2 / SIGMA) * (grad_nu_tilde * grad_nu_tilde).sum(axis=1) * mesh.cell_V_C


def update_nu_eff_f(mesh, nu_eff_f, nu_t, nu_t_f, nu_tilde, nu, u, v, wall_treatment):
    compute_nu_t(nu_t, nu_tilde, nu)
    interpolate_face(mesh, nu_t_f, nu_t)
    nu_eff_f[:] = nu + nu_t_f
    if wall_treatment == "integrate_to_wall":
        nu_eff_f[mesh.face_is_body] = nu
    else:
        nu_eff_f[mesh.face_is_body] = compute_nu_wall(mesh, u, v, nu)


def compute_nu_wall(mesh, u, v, nu):
    body = mesh.face_is_body
    body_C = mesh.face_C[body]
    y_c = mesh.face_d_Cf_mag[body]
    S_f = mesh.face_S_f[body]

    S_mag2 = (S_f * S_f).sum(axis=1)
    u_dot_S = u[body_C] * S_f[:, 0] + v[body_C] * S_f[:, 1]
    u_mag2 = u[body_C] ** 2 + v[body_C] ** 2
    u_par = np.sqrt(np.maximum(u_mag2 - u_dot_S * u_dot_S / S_mag2, 0.0))

    nu_wall = np.full_like(u_par, nu)
    valid = u_par > 1e-12
    if not valid.any():
        return nu_wall

    u_p = u_par[valid]
    y = y_c[valid]
    u_tau = np.sqrt(nu * u_p / y)

    for _ in range(MAX_ITER):
        u_plus = u_p / u_tau
        ku_plus = np.minimum(KAPPA * u_plus, KU_PLUS_MAX)
        y_plus = u_tau * y / nu
        exp_term = np.exp(ku_plus)
        bracket = exp_term - 1.0 - ku_plus - 0.5 * ku_plus**2 - ku_plus**3 / 6.0
        bracket_red = exp_term - 1.0 - ku_plus - 0.5 * ku_plus**2

        f = y_plus - u_plus - bracket / E
        f_prime = y / nu + u_p / u_tau**2 + ku_plus * bracket_red / (E * u_tau)
        du = f / f_prime
        u_tau = np.maximum(u_tau - du, 1e-10)
        if np.all(np.abs(du) < RTOL * u_tau):
            break

    nu_wall[valid] = u_tau * u_tau * y / u_p
    return nu_wall
