import gmsh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from dataclasses import dataclass, field

Y_1 = 1e-3

R_FAR = 30.0
H_FAR = 2.0

R_GROWTH = 1.2
R_OUT = 30.0
LE_LENGTH = 0.05
TE_THICKNESS = 0.05
N_AIRFOIL = 100
N_WAKE = 100
N_LE = 25


@dataclass
class Face:
    nodes: tuple[int, ...] = ()
    cells: tuple[int, ...] = ()

    x_f: np.ndarray | None = None
    S_f: np.ndarray | None = None
    S_f_mag: float = 0.0

    w_f: float = 0.0
    d_ff: np.ndarray | None = None

    d_CN_mag: float = 0.0
    d_CN_hat: np.ndarray | None = None

    d_Cf: np.ndarray | None = None
    d_Cf_mag: float = 0.0
    d_Nf: np.ndarray | None = None

    T_f: np.ndarray | None = None
    E_f_mag: float = 0.0


@dataclass
class Cell:
    nodes: tuple[int, ...] = ()
    faces: tuple[int, ...] = ()
    cells: tuple[int, ...] = ()
    signs: tuple[int, ...] = ()

    x_C: np.ndarray | None = None
    V_C: float = 0.0

    M_C_inv: np.ndarray | None = None


@dataclass
class Mesh:
    nodes: np.ndarray | None = None
    cells: tuple[Cell, ...] = ()
    faces: tuple[Face, ...] = ()
    face_tags: dict[str, tuple[int, ...]] = field(default_factory=dict)
    # {"interior": (...), "body": (...), "farfield": (...)}


def build_mesh(config) -> Mesh:
    pts = _read_body(config.body_file)
    pts = _scale_body(pts)
    pts = _canonicalise_body(pts)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("body")

    if config.mesh_type == "o_mesh":
        _generate_o_mesh(pts)
    elif config.mesh_type == "c_mesh":
        _generate_c_mesh(pts)
    else:
        gmsh.finalize()
        raise ValueError(f"unknown mesh type: {config.mesh_type}")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    mesh = Mesh()
    _load_mesh(mesh)
    _compute_face_connectivity(mesh)
    _compute_cell_connectivity(mesh)
    _compute_cell_geometry(mesh, config.gradient_type)
    _compute_face_geometry(mesh, config.diffusion_type)
    _compute_arrays(mesh)
    _compute_wall_distance(mesh)

    gmsh.finalize()

    return mesh


def _read_body(path):
    pts = []
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            x_str, y_str = line.split()
            pts.append((float(x_str), float(y_str)))

    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]

    if len(pts) < 3:
        raise ValueError(f"body must have at least 3 points, got {len(pts)}")

    return pts


def _scale_body(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    chord = x_max - x_min
    if chord == 0.0:
        raise ValueError("body has zero chord length")
    y_mid = 0.5 * (min(ys) + max(ys))
    return [((x - x_min) / chord, (y - y_mid) / chord) for (x, y) in pts]


def _canonicalise_body(pts):
    n = len(pts)
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    if s < 0.0:  # clockwise
        pts = list(reversed(pts))

    i_start = max(range(n), key=lambda i: pts[i][0])
    return pts[i_start:] + pts[:i_start]


def _generate_o_mesh(pts):
    geo = gmsh.model.geo

    body_pt_tags = [geo.addPoint(x, y, 0.0) for (x, y) in pts]
    body_pt_tags.append(body_pt_tags[0])
    body_curve = geo.addBSpline(body_pt_tags)
    body_loop = geo.addCurveLoop([body_curve])

    cx, cy = 0.5, 0.0
    centre = geo.addPoint(cx, cy, 0.0)
    p_e = geo.addPoint(cx + R_FAR, cy, 0.0)
    p_n = geo.addPoint(cx, cy + R_FAR, 0.0)
    p_w = geo.addPoint(cx - R_FAR, cy, 0.0)
    p_s = geo.addPoint(cx, cy - R_FAR, 0.0)
    arc1 = geo.addCircleArc(p_e, centre, p_n)
    arc2 = geo.addCircleArc(p_n, centre, p_w)
    arc3 = geo.addCircleArc(p_w, centre, p_s)
    arc4 = geo.addCircleArc(p_s, centre, p_e)
    far_loop = geo.addCurveLoop([arc1, arc2, arc3, arc4])

    surface = geo.addPlaneSurface([far_loop, body_loop])

    geo.synchronize()

    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", [body_curve])
    gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 200)

    f_thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", Y_1)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", H_FAR)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", R_FAR / 2.0)

    gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    pg_body = gmsh.model.addPhysicalGroup(1, [body_curve])
    gmsh.model.setPhysicalName(1, pg_body, "body")
    pg_far = gmsh.model.addPhysicalGroup(1, [arc1, arc2, arc3, arc4])
    gmsh.model.setPhysicalName(1, pg_far, "farfield")
    pg_domain = gmsh.model.addPhysicalGroup(2, [surface])
    gmsh.model.setPhysicalName(2, pg_domain, "domain")


def _generate_c_mesh(pts):
    # reference: https://github.com/Mikekiely/wuFoil/blob/master/wuFoil/meshing.py

    geo = gmsh.model.geo

    n = len(pts)
    le_idx = [i for i in range(n) if pts[i][0] <= LE_LENGTH]
    if len(le_idx) >= 2:
        i_top = le_idx[0]
        i_bot = le_idx[-1]
    else:
        i_top = i_bot = min(range(n), key=lambda i: pts[i][0])

    upper_pts = pts[: i_top + 1]
    le_pts = pts[i_top : i_bot + 1]
    lower_pts = pts[i_bot:] + [pts[0]]

    boundary_growth = R_GROWTH
    n_volume = int(round(np.log1p(R_OUT * (R_GROWTH - 1) / Y_1) / np.log(R_GROWTH))) + 1
    te_growth = (TE_THICKNESS / 0.1) ** (1.0 / (N_AIRFOIL - 1))
    wake_growth = 1.0 / np.exp(np.log(TE_THICKNESS) / (N_WAKE - 1))

    upper_tags = [geo.addPoint(x, y, 0.0) for (x, y) in upper_pts]
    af_top_tag = upper_tags[-1]
    le_tags = [af_top_tag] + [geo.addPoint(x, y, 0.0) for (x, y) in le_pts[1:]]
    af_bot_tag = le_tags[-1]
    lower_tags = [af_bot_tag] + [geo.addPoint(x, y, 0.0) for (x, y) in lower_pts[1:-1]]
    lower_tags.append(upper_tags[0])
    te_tag = upper_tags[0]

    af_upper = _add_body_curve(upper_tags)
    af_le = _add_body_curve(le_tags)
    af_lower = _add_body_curve(lower_tags)

    cx = pts[i_top][0]
    te_x = pts[0][0]
    out_x = te_x + R_OUT

    inlet_top = geo.addPoint(cx, R_OUT, 0.0)
    inlet_bot = geo.addPoint(cx, -R_OUT, 0.0)
    inlet_left = geo.addPoint(cx - R_OUT, 0.0, 0.0)
    centre = geo.addPoint(cx, 0.0, 0.0)
    top_te = geo.addPoint(te_x, R_OUT, 0.0)
    bot_te = geo.addPoint(te_x, -R_OUT, 0.0)
    wake_top = geo.addPoint(out_x, R_OUT, 0.0)
    wake_bot = geo.addPoint(out_x, -R_OUT, 0.0)
    wake_te = geo.addPoint(out_x, 0.0, 0.0)

    arc_upper = geo.addCircleArc(inlet_top, centre, inlet_left)
    arc_lower = geo.addCircleArc(inlet_left, centre, inlet_bot)
    afTop_inletTop = geo.addLine(af_top_tag, inlet_top)
    inletBot_afBot = geo.addLine(inlet_bot, af_bot_tag)
    topTe_afTe = geo.addLine(top_te, te_tag)
    afTe_botTe = geo.addLine(te_tag, bot_te)
    top_line = geo.addLine(top_te, inlet_top)
    bottom_line = geo.addLine(inlet_bot, bot_te)
    top_wake_line = geo.addLine(wake_top, top_te)
    bottom_wake_line = geo.addLine(bot_te, wake_bot)
    center_wake_line = geo.addLine(te_tag, wake_te)
    outlet_top = geo.addLine(wake_te, wake_top)
    outlet_bot = geo.addLine(wake_bot, wake_te)

    le_loop = geo.addCurveLoop(
        [
            afTop_inletTop,
            arc_upper,
            arc_lower,
            inletBot_afBot,
            -af_le,
        ]
    )
    le_surface = geo.addPlaneSurface([le_loop])

    top_loop = geo.addCurveLoop(
        [
            af_upper,
            afTop_inletTop,
            -top_line,
            topTe_afTe,
        ]
    )
    top_surface = geo.addPlaneSurface([top_loop])

    bot_loop = geo.addCurveLoop(
        [
            bottom_line,
            -afTe_botTe,
            -af_lower,
            -inletBot_afBot,
        ]
    )
    bot_surface = geo.addPlaneSurface([bot_loop])

    top_wake_loop = geo.addCurveLoop(
        [
            center_wake_line,
            outlet_top,
            top_wake_line,
            topTe_afTe,
        ]
    )
    top_wake_surface = geo.addPlaneSurface([top_wake_loop])

    bot_wake_loop = geo.addCurveLoop(
        [
            bottom_wake_line,
            outlet_bot,
            -center_wake_line,
            afTe_botTe,
        ]
    )
    bot_wake_surface = geo.addPlaneSurface([bot_wake_loop])

    mesh = geo.mesh
    mesh.setTransfiniteCurve(af_upper, N_AIRFOIL, "Progression", -te_growth)
    mesh.setTransfiniteCurve(af_lower, N_AIRFOIL, "Progression", te_growth)
    mesh.setTransfiniteCurve(af_le, 2 * N_LE - 1)
    mesh.setTransfiniteCurve(top_line, N_AIRFOIL, "Progression", -te_growth)
    mesh.setTransfiniteCurve(bottom_line, N_AIRFOIL, "Progression", -te_growth)
    mesh.setTransfiniteCurve(afTop_inletTop, n_volume, "Progression", boundary_growth)
    mesh.setTransfiniteCurve(inletBot_afBot, n_volume, "Progression", -boundary_growth)
    mesh.setTransfiniteCurve(topTe_afTe, n_volume, "Progression", -boundary_growth)
    mesh.setTransfiniteCurve(afTe_botTe, n_volume, "Progression", boundary_growth)
    mesh.setTransfiniteCurve(arc_upper, N_LE)
    mesh.setTransfiniteCurve(arc_lower, N_LE)
    mesh.setTransfiniteCurve(center_wake_line, N_WAKE, "Progression", wake_growth)
    mesh.setTransfiniteCurve(top_wake_line, N_WAKE, "Progression", -wake_growth)
    mesh.setTransfiniteCurve(bottom_wake_line, N_WAKE, "Progression", wake_growth)
    mesh.setTransfiniteCurve(outlet_top, n_volume, "Progression", boundary_growth)
    mesh.setTransfiniteCurve(outlet_bot, n_volume, "Progression", -boundary_growth)

    mesh.setTransfiniteSurface(
        le_surface, "Left", [af_top_tag, inlet_top, inlet_bot, af_bot_tag]
    )
    mesh.setTransfiniteSurface(top_surface)
    mesh.setTransfiniteSurface(bot_surface)
    mesh.setTransfiniteSurface(top_wake_surface)
    mesh.setTransfiniteSurface(bot_wake_surface)

    for s in (le_surface, top_surface, bot_surface, top_wake_surface, bot_wake_surface):
        mesh.setRecombine(2, s)

    pg_body = gmsh.model.addPhysicalGroup(1, [af_upper, af_le, af_lower])
    gmsh.model.setPhysicalName(1, pg_body, "body")
    pg_far = gmsh.model.addPhysicalGroup(
        1,
        [
            arc_upper,
            arc_lower,
            top_line,
            bottom_line,
            top_wake_line,
            bottom_wake_line,
            outlet_top,
            outlet_bot,
        ],
    )
    gmsh.model.setPhysicalName(1, pg_far, "farfield")
    pg_domain = gmsh.model.addPhysicalGroup(
        2,
        [le_surface, top_surface, bot_surface, top_wake_surface, bot_wake_surface],
    )
    gmsh.model.setPhysicalName(2, pg_domain, "domain")


def _add_body_curve(pt_tags):
    if len(pt_tags) == 2:
        return gmsh.model.geo.addLine(pt_tags[0], pt_tags[1])
    return gmsh.model.geo.addBSpline(pt_tags)


def _load_mesh(mesh):
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    n_nodes = len(node_tags)
    coords = np.array(node_coords).reshape(n_nodes, 3)[:, :2]
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
    mesh.nodes = coords

    cells = []
    elem_types, elem_tags_per_type, node_tags_per_type = gmsh.model.mesh.getElements(
        dim=2
    )
    for _, tags, ntags in zip(elem_types, elem_tags_per_type, node_tags_per_type):
        n_per_elem = len(ntags) // len(tags)
        for k in range(len(tags)):
            cell_node_tags = ntags[k * n_per_elem : (k + 1) * n_per_elem]
            cell_nodes = tuple(tag_to_idx[int(t)] for t in cell_node_tags)
            cells.append(Cell(nodes=cell_nodes))
    mesh.cells = tuple(cells)

    edge_to_face_idx = {}
    faces = []
    for cell in mesh.cells:
        n = len(cell.nodes)
        for k in range(n):
            a, b = cell.nodes[k], cell.nodes[(k + 1) % n]
            key = (a, b) if a < b else (b, a)
            if key not in edge_to_face_idx:
                edge_to_face_idx[key] = len(faces)
                faces.append(Face(nodes=key))
    mesh.faces = tuple(faces)

    body_idx = []
    far_idx = []
    for dim, tag in gmsh.model.getPhysicalGroups(dim=1):
        name = gmsh.model.getPhysicalName(dim, tag)
        ent_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        for ent in ent_tags:
            _, _, edge_node_tags = gmsh.model.mesh.getElements(dim=1, tag=int(ent))
            for ntags in edge_node_tags:
                for k in range(0, len(ntags), 2):
                    a = tag_to_idx[int(ntags[k])]
                    b = tag_to_idx[int(ntags[k + 1])]
                    key = (a, b) if a < b else (b, a)
                    fi = edge_to_face_idx[key]
                    if name == "body":
                        body_idx.append(fi)
                    elif name == "farfield":
                        far_idx.append(fi)

    body_set = set(body_idx)
    far_set = set(far_idx)
    interior_idx = [
        i for i in range(len(faces)) if i not in body_set and i not in far_set
    ]

    mesh.face_tags = {
        "interior": tuple(interior_idx),
        "body": tuple(sorted(body_set)),
        "farfield": tuple(sorted(far_set)),
    }


def _compute_face_connectivity(mesh):
    edge_to_face_idx = {face.nodes: fi for fi, face in enumerate(mesh.faces)}

    cell_faces = [[] for _ in mesh.cells]
    face_cells = [[] for _ in mesh.faces]

    for ci, cell in enumerate(mesh.cells):
        n = len(cell.nodes)
        for k in range(n):
            a, b = cell.nodes[k], cell.nodes[(k + 1) % n]
            key = (a, b) if a < b else (b, a)
            fi = edge_to_face_idx[key]
            cell_faces[ci].append(fi)
            face_cells[fi].append(ci)

    new_cells = []
    for cell, faces in zip(mesh.cells, cell_faces):
        new_cells.append(Cell(nodes=cell.nodes, faces=tuple(faces)))
    mesh.cells = tuple(new_cells)

    new_faces = []
    for face, cells in zip(mesh.faces, face_cells):
        new_faces.append(Face(nodes=face.nodes, cells=tuple(cells)))
    mesh.faces = tuple(new_faces)


def _compute_cell_connectivity(mesh):
    new_cells = []
    for ci, cell in enumerate(mesh.cells):
        signs = []
        neighbours = []
        for fi in cell.faces:
            face = mesh.faces[fi]
            if face.cells[0] == ci:
                signs.append(1)
                if len(face.cells) == 2:
                    neighbours.append(face.cells[1])
            else:
                signs.append(-1)
                neighbours.append(face.cells[0])
        new_cells.append(
            Cell(
                nodes=cell.nodes,
                faces=cell.faces,
                cells=tuple(neighbours),
                signs=tuple(signs),
            )
        )
    mesh.cells = tuple(new_cells)


def _compute_cell_geometry(mesh, gradient_type):
    new_cells = []
    for cell in mesh.cells:
        pts = mesh.nodes[list(cell.nodes)]
        n = len(pts)
        a2 = 0.0
        cx = 0.0
        cy = 0.0
        for k in range(n):
            x1, y1 = pts[k]
            x2, y2 = pts[(k + 1) % n]
            cross = x1 * y2 - x2 * y1
            a2 += cross
            cx += (x1 + x2) * cross
            cy += (y1 + y2) * cross
        V_C = 0.5 * abs(a2)
        x_C = np.array([cx / (3.0 * a2), cy / (3.0 * a2)])
        new_cells.append(
            Cell(
                nodes=cell.nodes,
                faces=cell.faces,
                cells=cell.cells,
                signs=cell.signs,
                x_C=x_C,
                V_C=V_C,
            )
        )

    if gradient_type == "least_squares":
        with_M = []
        for cell in new_cells:
            M = np.zeros((2, 2))
            for j in cell.cells:
                d = new_cells[j].x_C - cell.x_C
                M += np.outer(d, d) / (d @ d)
            M_inv = np.linalg.inv(M)
            with_M.append(
                Cell(
                    nodes=cell.nodes,
                    faces=cell.faces,
                    cells=cell.cells,
                    signs=cell.signs,
                    x_C=cell.x_C,
                    V_C=cell.V_C,
                    M_C_inv=M_inv,
                )
            )
        new_cells = with_M

    mesh.cells = tuple(new_cells)


def _compute_face_geometry(mesh, diffusion_type):
    new_faces = []
    for face in mesh.faces:
        a_idx, b_idx = face.nodes
        a = mesh.nodes[a_idx]
        b = mesh.nodes[b_idx]
        x_f = 0.5 * (a + b)
        edge = b - a
        S_f = np.array([edge[1], -edge[0]])

        C = face.cells[0]
        x_C = mesh.cells[C].x_C
        if S_f @ (x_f - x_C) < 0:
            S_f = -S_f

        S_f_mag = float(np.linalg.norm(S_f))
        d_Cf = x_f - x_C
        d_Cf_mag = float(np.linalg.norm(d_Cf))

        if len(face.cells) == 2:
            N = face.cells[1]
            x_N = mesh.cells[N].x_C
            d_Nf = x_f - x_N
            CN = x_N - x_C
            d_CN_mag = float(np.linalg.norm(CN))
            e_hat = CN / d_CN_mag
            d_CN_hat = e_hat
            w_f = float(
                np.linalg.norm(d_Nf) / (np.linalg.norm(d_Cf) + np.linalg.norm(d_Nf))
            )
            x_fp = x_C + (d_Cf @ e_hat) * e_hat
            d_ff = x_f - x_fp
        else:
            d_Nf = None
            d_CN_mag = 0.0
            d_CN_hat = None
            e_hat = d_Cf / d_Cf_mag
            w_f = 1.0
            d_ff = np.zeros(2)

        if diffusion_type == "minimum":
            E_f = (S_f @ e_hat) * e_hat
        elif diffusion_type == "orthogonal":
            E_f = S_f_mag * e_hat
        elif diffusion_type == "over_relaxed":
            E_f = (S_f_mag**2 / (S_f @ e_hat)) * e_hat
        else:
            raise ValueError(f"unknown diffusion type: {diffusion_type}")
        T_f = S_f - E_f
        E_f_mag = float(np.linalg.norm(E_f))

        new_faces.append(
            Face(
                nodes=face.nodes,
                cells=face.cells,
                x_f=x_f,
                S_f=S_f,
                S_f_mag=S_f_mag,
                w_f=w_f,
                d_ff=d_ff,
                d_CN_mag=d_CN_mag,
                d_CN_hat=d_CN_hat,
                d_Cf=d_Cf,
                d_Cf_mag=d_Cf_mag,
                d_Nf=d_Nf,
                T_f=T_f,
                E_f_mag=E_f_mag,
            )
        )
    mesh.faces = tuple(new_faces)


def _compute_arrays(mesh):
    import scipy.sparse as sp

    n_faces = len(mesh.faces)
    n_cells = len(mesh.cells)

    mesh.face_C = np.array([f.cells[0] for f in mesh.faces], dtype=np.int64)
    mesh.face_N = np.array(
        [f.cells[1] if len(f.cells) == 2 else -1 for f in mesh.faces], dtype=np.int64
    )
    mesh.face_x_f = np.array([f.x_f for f in mesh.faces])
    mesh.face_w_f = np.array([f.w_f for f in mesh.faces])
    mesh.face_S_f = np.array([f.S_f for f in mesh.faces])
    mesh.face_T_f = np.array([f.T_f for f in mesh.faces])
    mesh.face_d_ff = np.array([f.d_ff for f in mesh.faces])
    mesh.face_d_Cf = np.array([f.d_Cf for f in mesh.faces])
    mesh.face_d_Nf = np.array(
        [f.d_Nf if f.d_Nf is not None else np.zeros(2) for f in mesh.faces]
    )
    mesh.face_E_f_mag = np.array([f.E_f_mag for f in mesh.faces])
    mesh.face_d_CN_mag = np.array(
        [f.d_CN_mag if f.d_CN_mag != 0.0 else 1.0 for f in mesh.faces]
    )
    mesh.face_d_CN_hat = np.array(
        [f.d_CN_hat if f.d_CN_hat is not None else np.zeros(2) for f in mesh.faces]
    )
    mesh.face_d_Cf_mag = np.array([f.d_Cf_mag for f in mesh.faces])

    mesh.face_is_body = np.zeros(n_faces, dtype=bool)
    mesh.face_is_body[list(mesh.face_tags["body"])] = True
    mesh.face_is_far = np.zeros(n_faces, dtype=bool)
    mesh.face_is_far[list(mesh.face_tags["farfield"])] = True
    mesh.face_is_interior = ~(mesh.face_is_body | mesh.face_is_far)

    mesh.cell_V_C = np.array([c.V_C for c in mesh.cells])
    mesh.cell_x_C = np.array([c.x_C for c in mesh.cells])

    max_faces = max(len(c.faces) for c in mesh.cells)
    cell_faces = np.full((n_cells, max_faces), -1, dtype=np.int64)
    cell_signs = np.zeros((n_cells, max_faces), dtype=np.int64)
    cell_cells = np.full((n_cells, max_faces), -1, dtype=np.int64)
    for ci, c in enumerate(mesh.cells):
        for k, fi in enumerate(c.faces):
            cell_faces[ci, k] = fi
            cell_signs[ci, k] = c.signs[k]
        for k, cj in enumerate(c.cells):
            cell_cells[ci, k] = cj
    mesh.cell_faces = cell_faces
    mesh.cell_signs = cell_signs
    mesh.cell_cells = cell_cells

    if mesh.cells[0].M_C_inv is not None:
        mesh.cell_M_C_inv = np.array([c.M_C_inv for c in mesh.cells])
    else:
        mesh.cell_M_C_inv = None

    K = 5.0
    mesh.cell_eps2 = (K * np.sqrt(np.abs(mesh.cell_V_C))) ** 3

    rows = []
    cols = []
    data_x = []
    data_y = []
    for ci, c in enumerate(mesh.cells):
        for k, fi in enumerate(c.faces):
            sign = c.signs[k]
            S_f = mesh.faces[fi].S_f
            rows.append(ci)
            cols.append(fi)
            data_x.append(sign * S_f[0] / c.V_C)
            data_y.append(sign * S_f[1] / c.V_C)
    mesh.M_grad_x = sp.csr_matrix((data_x, (rows, cols)), shape=(n_cells, n_faces))
    mesh.M_grad_y = sp.csr_matrix((data_y, (rows, cols)), shape=(n_cells, n_faces))


def _compute_wall_distance(mesh):
    body_idx = np.where(mesh.face_is_body)[0]
    n_cells = len(mesh.cells)

    if len(body_idx) == 0:
        mesh.cell_d = np.full(n_cells, np.inf)
        return

    a = np.array([mesh.nodes[mesh.faces[fi].nodes[0]] for fi in body_idx])
    b = np.array([mesh.nodes[mesh.faces[fi].nodes[1]] for fi in body_idx])
    ab = b - a
    L2 = np.maximum((ab * ab).sum(axis=1), 1e-30)

    x = mesh.cell_x_C
    ap = x[:, None, :] - a[None, :, :]
    t = np.clip((ap * ab[None]).sum(axis=2) / L2[None], 0.0, 1.0)
    closest = a[None] + t[..., None] * ab[None]
    diff = x[:, None, :] - closest
    d = np.sqrt((diff * diff).sum(axis=2))
    mesh.cell_d = d.min(axis=1)


def show_mesh(mesh):
    from matplotlib.path import Path

    fig, ax = plt.subplots(figsize=(10, 10))

    polys = [mesh.nodes[list(cell.nodes)] for cell in mesh.cells]
    pc = PolyCollection(
        polys, facecolors="white", edgecolors="lightgrey", linewidths=0.3
    )
    ax.add_collection(pc)

    for tag_name, colour in (("body", "red"), ("farfield", "blue")):
        for fi in mesh.face_tags.get(tag_name, ()):
            face = mesh.faces[fi]
            seg = mesh.nodes[list(face.nodes)]
            ax.plot(seg[:, 0], seg[:, 1], color=colour, linewidth=0.8)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_title(
        f"{len(mesh.cells)} cells, {len(mesh.faces)} faces, {len(mesh.nodes)} nodes"
    )

    cell_paths = [Path(p) for p in polys]
    body_set = set(mesh.face_tags.get("body", ()))
    far_set = set(mesh.face_tags.get("farfield", ()))
    selection = []

    def fmt_vec(v):
        if v is None:
            return "None"
        return f"[{v[0]:+.4g}, {v[1]:+.4g}]"

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        xy = np.array([event.xdata, event.ydata])

        ci = next((i for i, p in enumerate(cell_paths) if p.contains_point(xy)), None)
        if ci is None:
            return

        cell = mesh.cells[ci]

        while selection:
            selection.pop().remove()

        cell_patch = PolyCollection(
            [polys[ci]],
            facecolors="lightgrey",
            edgecolors="none",
            alpha=0.4,
        )
        ax.add_collection(cell_patch)
        selection.append(cell_patch)

        (dot_C,) = ax.plot(cell.x_C[0], cell.x_C[1], "o", color="blue", markersize=6)
        selection.append(dot_C)

        L_scale = 1.5 * np.sqrt(abs(cell.V_C))
        for k, fi in enumerate(cell.faces):
            face = mesh.faces[fi]
            seg = mesh.nodes[list(face.nodes)]
            (face_line,) = ax.plot(seg[:, 0], seg[:, 1], color="dimgrey", linewidth=2.5)
            selection.append(face_line)

            (dot_f,) = ax.plot(
                face.x_f[0], face.x_f[1], "o", color="green", markersize=6
            )
            selection.append(dot_f)

            S = cell.signs[k] * face.S_f
            L = L_scale / face.S_f_mag
            arrow = ax.annotate(
                "",
                xy=(face.x_f[0] + S[0] * L, face.x_f[1] + S[1] * L),
                xytext=(face.x_f[0], face.x_f[1]),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.3),
            )
            selection.append(arrow)

            x_C0 = mesh.cells[face.cells[0]].x_C
            d_Cf_arrow = ax.annotate(
                "",
                xy=(face.x_f[0], face.x_f[1]),
                xytext=(x_C0[0], x_C0[1]),
                arrowprops=dict(arrowstyle="->", color="orange", lw=1.3),
            )
            selection.append(d_Cf_arrow)

            if len(face.cells) == 2:
                x_N = mesh.cells[face.cells[1]].x_C
                d_Nf_arrow = ax.annotate(
                    "",
                    xy=(face.x_f[0], face.x_f[1]),
                    xytext=(x_N[0], x_N[1]),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.3),
                )
                selection.append(d_Nf_arrow)

        ax.set_title(f"cell {ci}")

        print(f"=== Cell {ci} ===")
        print(f"  nodes:    {cell.nodes}")
        print(f"  faces:    {cell.faces}")
        print(f"  cells:    {cell.cells}")
        print(f"  signs:    {cell.signs}")
        print(f"  x_C:      {fmt_vec(cell.x_C)}")
        print(f"  V_C:      {cell.V_C:+.4g}")
        print(f"  M_C_inv:  {'set' if cell.M_C_inv is not None else 'None'}")
        for k, fi in enumerate(cell.faces):
            face = mesh.faces[fi]
            boundary = (
                "body"
                if fi in body_set
                else "farfield"
                if fi in far_set
                else "interior"
            )
            sign = cell.signs[k]
            print(f"--- Face {fi} ({boundary}, sign {sign:+d}) ---")
            print(f"  nodes:    {face.nodes}")
            print(f"  cells:    {face.cells}")
            print(f"  x_f:      {fmt_vec(face.x_f)}")
            print(f"  S_f:      {fmt_vec(face.S_f)}  |S_f| = {face.S_f_mag:+.4g}")
            print(f"  w_f:      {face.w_f:+.4g}")
            print(f"  d_ff:     {fmt_vec(face.d_ff)}")
            print(f"  d_CN_mag: {face.d_CN_mag:+.4g}")
            print(f"  d_CN_hat: {fmt_vec(face.d_CN_hat)}")
            print(f"  d_Cf:     {fmt_vec(face.d_Cf)}  |d_Cf| = {face.d_Cf_mag:+.4g}")
            print(f"  d_Nf:     {fmt_vec(face.d_Nf)}")
            print(f"  T_f:      {fmt_vec(face.T_f)}")
            print(f"  E_f_mag:  {face.E_f_mag:+.4g}")
        print()

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
