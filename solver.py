import numpy as np
import scipy.sparse as sp
from pathlib import Path
from collections import namedtuple

Coo = namedtuple("Coo", ["data", "rows", "cols"])


def build_sparse(A_coo, n):
    data = np.concatenate(A_coo.data) if A_coo.data else np.zeros(0)
    rows = np.concatenate(A_coo.rows) if A_coo.rows else np.zeros(0, dtype=np.int64)
    cols = np.concatenate(A_coo.cols) if A_coo.cols else np.zeros(0, dtype=np.int64)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def assemble_transport_A(mesh, A_coo, gamma_f, m_f, dt, is_outflow):
    n_cells = len(mesh.cells)

    A_coo.data.append(mesh.cell_V_C / dt)
    A_coo.rows.append(np.arange(n_cells, dtype=np.int64))
    A_coo.cols.append(np.arange(n_cells, dtype=np.int64))

    interior = mesh.face_is_interior
    body = mesh.face_is_body
    far = mesh.face_is_far
    inflow = far & ~is_outflow
    outflow = far & is_outflow

    if interior.any():
        C = mesh.face_C[interior]
        N = mesh.face_N[interior]

        D_f = (
            gamma_f[interior]
            * mesh.face_E_f_mag[interior]
            / mesh.face_d_CN_mag[interior]
        )
        A_coo.data.append(np.concatenate([D_f, -D_f, D_f, -D_f]))
        A_coo.rows.append(np.concatenate([C, C, N, N]))
        A_coo.cols.append(np.concatenate([C, N, N, C]))

        m_plus = np.maximum(m_f[interior], 0.0)
        m_minus = np.minimum(m_f[interior], 0.0)
        A_coo.data.append(np.concatenate([m_plus, m_minus, -m_minus, -m_plus]))
        A_coo.rows.append(np.concatenate([C, C, N, N]))
        A_coo.cols.append(np.concatenate([C, N, N, C]))

    if body.any():
        C = mesh.face_C[body]

        D_f = gamma_f[body] * mesh.face_E_f_mag[body] / mesh.face_d_Cf_mag[body]
        A_coo.data.append(D_f)
        A_coo.rows.append(C)
        A_coo.cols.append(C)

    if inflow.any():
        C = mesh.face_C[inflow]

        D_f = gamma_f[inflow] * mesh.face_E_f_mag[inflow] / mesh.face_d_Cf_mag[inflow]
        A_coo.data.append(D_f)
        A_coo.rows.append(C)
        A_coo.cols.append(C)

    if outflow.any():
        C = mesh.face_C[outflow]

        m_plus = np.maximum(m_f[outflow], 0.0)
        A_coo.data.append(m_plus)
        A_coo.rows.append(C)
        A_coo.cols.append(C)


def assemble_transport_b(
    mesh,
    b,
    phi,
    phi_f,
    grad_phi,
    grad_phi_f,
    m_f,
    gamma_f,
    dt,
    convection_type,
    is_outflow,
):
    psi = _compute_limiter(mesh, phi, grad_phi, convection_type)

    b += (mesh.cell_V_C / dt) * phi

    T_term = gamma_f * (grad_phi_f * mesh.face_T_f).sum(axis=1)

    interior = mesh.face_is_interior
    body = mesh.face_is_body
    far = mesh.face_is_far
    inflow = far & ~is_outflow
    outflow = far & is_outflow

    if interior.any():
        C = mesh.face_C[interior]
        N = mesh.face_N[interior]

        np.add.at(b, C, T_term[interior])
        np.add.at(b, N, -T_term[interior])

        m = m_f[interior]
        upwind_is_C = m >= 0.0
        U = np.where(upwind_is_C, C, N)
        d_Uf = np.where(
            upwind_is_C[:, None],
            mesh.face_d_Cf[interior],
            mesh.face_d_Nf[interior],
        )
        flux = m * psi[U] * (grad_phi[U] * d_Uf).sum(axis=1)
        np.add.at(b, C, -flux)
        np.add.at(b, N, flux)

    if body.any():
        C = mesh.face_C[body]

        D_f = gamma_f[body] * mesh.face_E_f_mag[body] / mesh.face_d_Cf_mag[body]
        np.add.at(b, C, D_f * phi_f[body] + T_term[body])

    if inflow.any():
        C = mesh.face_C[inflow]

        D_f = gamma_f[inflow] * mesh.face_E_f_mag[inflow] / mesh.face_d_Cf_mag[inflow]
        m_minus = np.minimum(m_f[inflow], 0.0)
        np.add.at(b, C, D_f * phi_f[inflow] + T_term[inflow] - m_minus * phi_f[inflow])

    if outflow.any():
        C = mesh.face_C[outflow]

        m_plus = np.maximum(m_f[outflow], 0.0)
        flux = m_plus * psi[C] * (grad_phi[C] * mesh.face_d_Cf[outflow]).sum(axis=1)
        np.add.at(b, C, -flux)


def _compute_limiter(mesh, phi, grad_phi, limiter_type):
    if limiter_type not in ("barth_jespersen", "venkatakrishnan"):
        raise ValueError(f"unknown limiter type: {limiter_type}")

    valid_n = mesh.cell_cells >= 0
    N_safe = np.where(valid_n, mesh.cell_cells, 0)
    nb_phi = phi[N_safe]
    phi_max = np.maximum(phi, np.where(valid_n, nb_phi, -np.inf).max(axis=1))
    phi_min = np.minimum(phi, np.where(valid_n, nb_phi, np.inf).min(axis=1))
    d_max = phi_max - phi
    d_min = phi_min - phi

    valid_f = mesh.cell_faces >= 0
    fi_safe = np.where(valid_f, mesh.cell_faces, 0)
    sign_pos = mesh.cell_signs[..., None] > 0
    d_Uf = np.where(sign_pos, mesh.face_d_Cf[fi_safe], mesh.face_d_Nf[fi_safe])

    d_f = (grad_phi[:, None, :] * d_Uf).sum(axis=-1)
    d_star = np.where(d_f > 0, d_max[:, None], d_min[:, None])

    if limiter_type == "barth_jespersen":
        with np.errstate(divide="ignore", invalid="ignore"):
            psi_candidates = np.minimum(1.0, d_star / d_f)
    else:
        eps2 = mesh.cell_eps2[:, None]
        num = d_star * d_star + 2.0 * d_f * d_star + eps2
        den = d_star * d_star + 2.0 * d_f * d_f + d_f * d_star + eps2
        with np.errstate(divide="ignore", invalid="ignore"):
            psi_candidates = num / den

    skip = (d_f == 0.0) | ~valid_f
    psi_candidates = np.where(skip, 1.0, psi_candidates)
    return psi_candidates.min(axis=1)


def add_pressure_source(mesh, b, source):
    b += mesh.cell_V_C * source


def assemble_pressure_A(mesh, A_coo, gamma_f, is_outflow):
    interior = mesh.face_is_interior
    far = mesh.face_is_far
    outflow = far & is_outflow

    if interior.any():
        C = mesh.face_C[interior]
        N = mesh.face_N[interior]

        D_f = (
            gamma_f[interior]
            * mesh.face_E_f_mag[interior]
            / mesh.face_d_CN_mag[interior]
        )
        A_coo.data.append(np.concatenate([D_f, -D_f, D_f, -D_f]))
        A_coo.rows.append(np.concatenate([C, C, N, N]))
        A_coo.cols.append(np.concatenate([C, N, N, C]))

    if outflow.any():
        C = mesh.face_C[outflow]

        D_f = (
            gamma_f[outflow] * mesh.face_E_f_mag[outflow] / mesh.face_d_Cf_mag[outflow]
        )
        A_coo.data.append(D_f)
        A_coo.rows.append(C)
        A_coo.cols.append(C)


def assemble_pressure_b(mesh, b, p_f, grad_p_f, gamma_f, m_f_star, is_outflow):
    interior = mesh.face_is_interior
    far = mesh.face_is_far
    inflow = far & ~is_outflow
    outflow = far & is_outflow

    T_term = gamma_f * (grad_p_f * mesh.face_T_f).sum(axis=1)

    if interior.any():
        C = mesh.face_C[interior]
        N = mesh.face_N[interior]

        np.add.at(b, C, -m_f_star[interior] + T_term[interior])
        np.add.at(b, N, m_f_star[interior] - T_term[interior])

    if inflow.any():
        C = mesh.face_C[inflow]

        np.add.at(b, C, -m_f_star[inflow])

    if outflow.any():
        C = mesh.face_C[outflow]

        np.add.at(b, C, -m_f_star[outflow] + T_term[outflow])


def compute_gamma(mesh, gamma, A):
    gamma[:] = mesh.cell_V_C / A.diagonal()


def compute_m_f_star(
    mesh, m_f_star, hx_f, hy_f, gamma_f, m_f_old, u_f_old, v_f_old, dt
):
    Sx = mesh.face_S_f[:, 0]
    Sy = mesh.face_S_f[:, 1]

    m_f_star[:] = hx_f * Sx + hy_f * Sy

    interior = mesh.face_is_interior
    if not interior.any():
        return
    u_dot_S = u_f_old[interior] * Sx[interior] + v_f_old[interior] * Sy[interior]
    m_f_star[interior] += (gamma_f[interior] / dt) * (m_f_old[interior] - u_dot_S)


def correct_velocity(mesh, u, v, hx, hy, gamma, grad_p):
    u[:] = hx - gamma * grad_p[:, 0]
    v[:] = hy - gamma * grad_p[:, 1]


def correct_mass_flux(mesh, m_f, m_f_star, gamma_f, p, grad_p_f, is_outflow):
    interior = mesh.face_is_interior
    body = mesh.face_is_body
    far = mesh.face_is_far
    inflow = far & ~is_outflow
    outflow = far & is_outflow

    m_f[body] = 0.0
    m_f[inflow] = m_f_star[inflow]

    correction_mask = interior | outflow
    if not correction_mask.any():
        return

    T_term = gamma_f * (grad_p_f * mesh.face_T_f).sum(axis=1)

    implicit = np.zeros_like(m_f)
    if interior.any():
        C = mesh.face_C[interior]
        N = mesh.face_N[interior]
        implicit[interior] = (
            gamma_f[interior]
            * mesh.face_E_f_mag[interior]
            * (p[N] - p[C])
            / mesh.face_d_CN_mag[interior]
        )
    if outflow.any():
        C = mesh.face_C[outflow]
        implicit[outflow] = (
            gamma_f[outflow]
            * mesh.face_E_f_mag[outflow]
            * (0.0 - p[C])
            / mesh.face_d_Cf_mag[outflow]
        )

    m_f[correction_mask] = (
        m_f_star[correction_mask] - implicit[correction_mask] - T_term[correction_mask]
    )


def compute_divergence(mesh, m_f, t):
    valid = mesh.cell_faces >= 0
    fi_safe = np.where(valid, mesh.cell_faces, 0)
    div = np.where(valid, mesh.cell_signs * m_f[fi_safe], 0.0).sum(axis=1)

    div_max = float(np.max(div))
    div_min = float(np.min(div))
    div_l1 = float(np.sum(np.abs(div)))
    div_l2 = float(np.sqrt(np.sum(div * div)))
    print(
        f"t={t:.6g}  div_max={div_max:+.4g}  div_min={div_min:+.4g}  "
        f"div_l1={div_l1:+.4g}  div_l2={div_l2:+.4g}"
    )


def write_data(mesh, p, u, v, w, nu_t, u_inf, v_inf, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    body = mesh.face_is_body
    xb = mesh.face_x_f[body, 0]
    yb = mesh.face_x_f[body, 1]
    pb = p[mesh.face_C[body]]

    x = mesh.cell_x_C[:, 0]
    y = mesh.cell_x_C[:, 1]

    np.savez(
        path,
        xb=xb,
        yb=yb,
        pb=pb,
        x=x,
        y=y,
        u=u,
        v=v,
        p=p,
        w=w,
        nu_t=nu_t,
        u_inf=u_inf,
        v_inf=v_inf,
    )
