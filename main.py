import math
import numpy as np
import scipy.sparse.linalg as spla
from dataclasses import dataclass

from mesh import (
    build_mesh,
    show_mesh,
)
from reconstruct import (
    interpolate_face,
    compute_gradient,
    reconstruct_face,
)
from solver import (
    Coo,
    build_sparse,
    assemble_transport_A,
    assemble_transport_b,
    add_pressure_source,
    assemble_pressure_A,
    assemble_pressure_b,
    compute_gamma,
    compute_m_f_star,
    correct_velocity,
    correct_mass_flux,
    compute_divergence,
    write_data,
)
from turbulence import (
    SIGMA,
    compute_closure,
    add_production,
    add_destruction,
    add_cross_diffusion,
    update_nu_eff_f,
)
from animate import animate_data


@dataclass
class Config:
    u_inf: float = math.cos(math.radians(5.0))
    v_inf: float = math.sin(math.radians(5.0))
    Re: float = 1e6
    dt: float = 1e-3
    nt: int = 20000
    n_piso: int = 2
    n_log: int = 1
    n_write: int = 100

    body_file: str = "body.dat"
    mesh_type: str = "c_mesh"  # o_mesh, c_mesh
    check_mesh: bool = False
    gradient_type: str = "green_gauss"  # green_gauss, least_squares
    diffusion_type: str = "over_relaxed"  # over_relaxed, minimum, orthogonal
    convection_type: str = "barth_jespersen"  # barth_jespersen, venkatakrishnan

    turbulence: bool = True
    wall_treatment: str = "wall_function"  # integrate_to_wall, wall_function
    chi_inf: float = 3.0

    n_refactor: int = 25
    solver_rtol: float = 1e-8
    solver_maxiter: int = 200


config = Config()

u_inf = config.u_inf
v_inf = config.v_inf
Re = config.Re
dt = config.dt
nt = config.nt
n_piso = config.n_piso
n_log = config.n_log
n_write = config.n_write
body_file = config.body_file
check_mesh = config.check_mesh
gradient_type = config.gradient_type
convection_type = config.convection_type
turbulence = config.turbulence
wall_treatment = config.wall_treatment
chi_inf = config.chi_inf
n_refactor = config.n_refactor
solver_rtol = config.solver_rtol
solver_maxiter = config.solver_maxiter

if wall_treatment not in ("integrate_to_wall", "wall_function"):
    raise ValueError(f"unknown wall_treatment: {wall_treatment}")

mesh = build_mesh(config)
if check_mesh:
    show_mesh(mesh)

n_cells = len(mesh.cells)
n_faces = len(mesh.faces)

u = np.zeros(n_cells)
u_f = np.zeros(n_faces)
grad_u = np.zeros((n_cells, 2))
grad_u_f = np.zeros((n_faces, 2))

v = np.zeros(n_cells)
v_f = np.zeros(n_faces)
grad_v = np.zeros((n_cells, 2))
grad_v_f = np.zeros((n_faces, 2))

p = np.zeros(n_cells)
p_f = np.zeros(n_faces)
grad_p = np.zeros((n_cells, 2))
grad_p_f = np.zeros((n_faces, 2))

hx = np.zeros(n_cells)
hx_f = np.zeros(n_faces)
grad_hx = np.zeros((n_cells, 2))
grad_hx_f = np.zeros((n_faces, 2))

hy = np.zeros(n_cells)
hy_f = np.zeros(n_faces)
grad_hy = np.zeros((n_cells, 2))
grad_hy_f = np.zeros((n_faces, 2))

m_f = np.zeros(n_faces)
m_f_star = np.zeros(n_faces)

m_f_old = np.zeros(n_faces)
u_f_old = np.zeros(n_faces)
v_f_old = np.zeros(n_faces)

b = np.zeros(n_cells)
gamma = np.zeros(n_cells)
gamma_f = np.zeros(n_faces)

nu = 1.0 / Re
nu_eff_f = np.full(n_faces, nu)

nu_tilde = np.zeros(n_cells)
nu_tilde_f = np.zeros(n_faces)
grad_nu_tilde = np.zeros((n_cells, 2))
grad_nu_tilde_f = np.zeros((n_faces, 2))
nu_t = np.zeros(n_cells)
nu_t_f = np.zeros(n_faces)
S_tilde = np.zeros(n_cells)
f_w = np.zeros(n_cells)

nu_tilde_inf = chi_inf * nu

is_outflow = np.zeros(n_faces, dtype=bool)


def apply_bc_u(u_f):
    u_f[mesh.face_is_body] = 0.0
    u_f[mesh.face_is_far & ~is_outflow] = u_inf


def apply_bc_v(v_f):
    v_f[mesh.face_is_body] = 0.0
    v_f[mesh.face_is_far & ~is_outflow] = v_inf


def apply_bc_p(p_f):
    p_f[mesh.face_is_far & is_outflow] = 0.0


def apply_bc_nu_tilde(nu_tilde_f):
    nu_tilde_f[mesh.face_is_body] = 0.0
    nu_tilde_f[mesh.face_is_far & ~is_outflow] = nu_tilde_inf


# initial conditions

t = 0.0

u[:] = u_inf
v[:] = v_inf
u_f[:] = u_inf
v_f[:] = v_inf

apply_bc_u(u_f)
apply_bc_v(v_f)

m_f[:] = mesh.face_S_f[:, 0] * u_f + mesh.face_S_f[:, 1] * v_f

if turbulence:
    nu_tilde[:] = nu_tilde_inf
    update_nu_eff_f(mesh, nu_eff_f, nu_t, nu_t_f, nu_tilde, nu, u, v, wall_treatment)

# cached preconditioners

A_lu = None
A_p_lu = None
A_tilde_lu = None

# time loop

for n in range(nt):
    t += dt

    m_f_old[:] = m_f

    # inflow/outflow classification

    is_outflow[:] = mesh.face_is_far & (m_f > 0.0)

    # velocity face reconstruction

    reconstruct_face(mesh, u, u_f, grad_u, grad_u_f, gradient_type, apply_bc_u)
    reconstruct_face(mesh, v, v_f, grad_v, grad_v_f, gradient_type, apply_bc_v)
    u_f_old[:] = u_f
    v_f_old[:] = v_f

    # build momentum matrix

    A_coo = Coo([], [], [])
    assemble_transport_A(mesh, A_coo, nu_eff_f, m_f, dt, is_outflow)
    A = build_sparse(A_coo, n_cells)
    if A_lu is None or n % n_refactor == 0:
        A_lu = spla.splu(A.tocsc())
    M = spla.LinearOperator(A.shape, matvec=A_lu.solve)

    # u-velocity predictor

    b[:] = 0.0
    assemble_transport_b(
        mesh,
        b,
        u,
        u_f,
        grad_u,
        grad_u_f,
        m_f,
        nu_eff_f,
        dt,
        convection_type,
        is_outflow,
    )
    b_u_no_p = b.copy()
    add_pressure_source(mesh, b, -grad_p[:, 0])
    u[:], _ = spla.bicgstab(
        A, b, M=M, x0=u.copy(), rtol=solver_rtol, maxiter=solver_maxiter
    )

    # v-velocity predictor

    b[:] = 0.0
    assemble_transport_b(
        mesh,
        b,
        v,
        v_f,
        grad_v,
        grad_v_f,
        m_f,
        nu_eff_f,
        dt,
        convection_type,
        is_outflow,
    )
    b_v_no_p = b.copy()
    add_pressure_source(mesh, b, -grad_p[:, 1])
    v[:], _ = spla.bicgstab(
        A, b, M=M, x0=v.copy(), rtol=solver_rtol, maxiter=solver_maxiter
    )

    # compute gamma_f

    compute_gamma(mesh, gamma, A)
    interpolate_face(mesh, gamma_f, gamma)
    a_diag = A.diagonal()

    # build pressure matrix

    A_coo = Coo([], [], [])
    assemble_pressure_A(mesh, A_coo, gamma_f, is_outflow)
    A_p = build_sparse(A_coo, n_cells)
    if A_p_lu is None or n % n_refactor == 0:
        A_p_lu = spla.splu(A_p.tocsc())
    M_p = spla.LinearOperator(A_p.shape, matvec=A_p_lu.solve)

    # pressure corrector (PISO loop)

    for inner in range(n_piso):
        # compute h_f

        hx[:] = u + (b_u_no_p - A @ u) / a_diag
        hy[:] = v + (b_v_no_p - A @ v) / a_diag

        reconstruct_face(mesh, hx, hx_f, grad_hx, grad_hx_f, gradient_type, apply_bc_u)
        reconstruct_face(mesh, hy, hy_f, grad_hy, grad_hy_f, gradient_type, apply_bc_v)

        # compute m_f_star

        compute_m_f_star(
            mesh, m_f_star, hx_f, hy_f, gamma_f, m_f_old, u_f_old, v_f_old, dt
        )

        # solve pressure

        b[:] = 0.0
        assemble_pressure_b(mesh, b, p_f, grad_p_f, gamma_f, m_f_star, is_outflow)
        p[:], _ = spla.cg(
            A_p, b, M=M_p, x0=p.copy(), rtol=solver_rtol, maxiter=solver_maxiter
        )

        # correct mass flux

        correct_mass_flux(mesh, m_f, m_f_star, gamma_f, p, grad_p_f, is_outflow)

        # pressure face reconstruction

        reconstruct_face(mesh, p, p_f, grad_p, grad_p_f, gradient_type, apply_bc_p)

        # correct velocity

        correct_velocity(mesh, u, v, hx, hy, gamma, grad_p)

    # turbulence equation

    if turbulence:
        # nu_tilde face reconstruction

        reconstruct_face(
            mesh,
            nu_tilde,
            nu_tilde_f,
            grad_nu_tilde,
            grad_nu_tilde_f,
            gradient_type,
            apply_bc_nu_tilde,
        )

        # compute closure

        Omega = np.abs(grad_v[:, 0] - grad_u[:, 1])
        gamma_tilde_f = (nu + nu_tilde_f) / SIGMA
        compute_closure(S_tilde, f_w, nu_tilde, nu, Omega, mesh.cell_d)

        # build turbulence matrix

        A_coo = Coo([], [], [])
        assemble_transport_A(mesh, A_coo, gamma_tilde_f, m_f, dt, is_outflow)
        add_destruction(mesh, A_coo, nu_tilde, f_w)
        A_tilde = build_sparse(A_coo, n_cells)
        if A_tilde_lu is None or n % n_refactor == 0:
            A_tilde_lu = spla.splu(A_tilde.tocsc())
        M_tilde = spla.LinearOperator(A_tilde.shape, matvec=A_tilde_lu.solve)

        # solve turbulence

        b[:] = 0.0
        assemble_transport_b(
            mesh,
            b,
            nu_tilde,
            nu_tilde_f,
            grad_nu_tilde,
            grad_nu_tilde_f,
            m_f,
            gamma_tilde_f,
            dt,
            convection_type,
            is_outflow,
        )
        add_production(mesh, b, S_tilde, nu_tilde)
        add_cross_diffusion(mesh, b, grad_nu_tilde)
        nu_tilde[:], _ = spla.bicgstab(
            A_tilde,
            b,
            M=M_tilde,
            x0=nu_tilde.copy(),
            rtol=solver_rtol,
            maxiter=solver_maxiter,
        )
        np.maximum(nu_tilde, 0.0, out=nu_tilde)

        # update nu_t and nu_eff_f

        update_nu_eff_f(
            mesh,
            nu_eff_f,
            nu_t,
            nu_t_f,
            nu_tilde,
            nu,
            u,
            v,
            wall_treatment,
        )

    if n % n_log == 0:
        compute_divergence(mesh, m_f, t)

    if n % n_write == 0:
        compute_gradient(mesh, grad_u, u, u_f, gradient_type)
        compute_gradient(mesh, grad_v, v, v_f, gradient_type)
        w = grad_v[:, 0] - grad_u[:, 1]  # vorticity
        write_data(mesh, p, u, v, w, nu_t, u_inf, v_inf, f"results/snapshot_{n}.npz")

animate_data(body_file)
