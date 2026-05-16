import numpy as np


def interpolate_face(mesh, phi_f, phi):
    phi_f[:] = phi[mesh.face_C]

    interior = mesh.face_is_interior
    if not interior.any():
        return
    C = mesh.face_C[interior]
    N = mesh.face_N[interior]
    w = mesh.face_w_f[interior]
    if phi.ndim == 1:
        phi_f[interior] = w * phi[C] + (1.0 - w) * phi[N]
    else:
        w = w[:, None]
        phi_f[interior] = w * phi[C] + (1.0 - w) * phi[N]


def compute_gradient(mesh, grad_phi, phi, phi_f, gradient_type):
    if gradient_type == "green_gauss":
        grad_phi[:, 0] = mesh.M_grad_x @ phi_f
        grad_phi[:, 1] = mesh.M_grad_y @ phi_f
    elif gradient_type == "least_squares":
        valid = mesh.cell_cells >= 0
        N_safe = np.where(valid, mesh.cell_cells, 0)

        d_CN = mesh.cell_x_C[N_safe] - mesh.cell_x_C[:, None, :]
        d_CN_sq = np.where(valid, (d_CN * d_CN).sum(axis=-1), 1.0)
        dphi = phi[N_safe] - phi[:, None]

        weight = np.where(valid, dphi / d_CN_sq, 0.0)[..., None]
        r_C = (weight * d_CN).sum(axis=1)
        grad_phi[:] = np.einsum("cij,cj->ci", mesh.cell_M_C_inv, r_C)
    else:
        raise ValueError(f"unknown gradient type: {gradient_type}")


def _correct_orthogonality(mesh, grad_phi_f, phi):
    interior = mesh.face_is_interior
    if not interior.any():
        return
    C = mesh.face_C[interior]
    N = mesh.face_N[interior]

    d_CN_mag = mesh.face_d_CN_mag[interior]
    d_CN_hat = mesh.face_d_CN_hat[interior]

    central = (phi[N] - phi[C]) / d_CN_mag
    aligned = (grad_phi_f[interior] * d_CN_hat).sum(axis=1)
    grad_phi_f[interior] += (central - aligned)[:, None] * d_CN_hat


def _correct_skewness(mesh, phi_f, grad_phi_f):
    phi_f += (grad_phi_f * mesh.face_d_ff).sum(axis=1)


def reconstruct_face(mesh, phi, phi_f, grad_phi, grad_phi_f, gradient_type, apply_bc):
    interpolate_face(mesh, phi_f, phi)
    apply_bc(phi_f)
    compute_gradient(mesh, grad_phi, phi, phi_f, gradient_type)
    interpolate_face(mesh, grad_phi_f, grad_phi)
    _correct_orthogonality(mesh, grad_phi_f, phi)
    _correct_skewness(mesh, phi_f, grad_phi_f)
    apply_bc(phi_f)
