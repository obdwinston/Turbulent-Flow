from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.tri as mtri

RESULTS_DIR = "results"
SAVE_PATH = f"{RESULTS_DIR}/animation.mp4"
VERIFICATION_PATH = "verification.dat"
FPS = 10

CP_XLIM = (-0.1, 1.1)
CP_YLIM = (-2.5, 1.5)
FIELD_XLIM = (-1.0, 3.0)
FIELD_YLIM = (-1.0, 1.0)
CLIP_FIELD = 95.0
CLIP_VORT = 95.0


def _load_body(path):
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
    pts = np.array(pts)
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    chord = x_max - x_min
    y_mid = 0.5 * (pts[:, 1].min() + pts[:, 1].max())
    body_x = (pts[:, 0] - x_min) / chord
    body_y = (pts[:, 1] - y_mid) / chord
    return np.append(body_x, body_x[0]), np.append(body_y, body_y[0])


def animate_data(body_file="body.dat"):
    files = sorted(
        Path(RESULTS_DIR).glob("snapshot_*.npz"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not files:
        raise FileNotFoundError(f"no snapshot_*.npz files in {RESULTS_DIR}")

    snapshots = []
    for f in files:
        d = np.load(f)
        snapshots.append({"name": f.stem, **{k: d[k] for k in d.files}})

    u_inf = float(snapshots[0]["u_inf"])
    v_inf = float(snapshots[0]["v_inf"])
    q = 0.5 * (u_inf * u_inf + v_inf * v_inf)

    triang = mtri.Triangulation(snapshots[0]["x"], snapshots[0]["y"])

    # body plot

    body_x, body_y = _load_body(body_file)

    lo, hi = 100 - CLIP_FIELD, CLIP_FIELD
    p_min, p_max = np.percentile(np.concatenate([s["p"] for s in snapshots]), [lo, hi])
    u_min, u_max = np.percentile(np.concatenate([s["u"] for s in snapshots]), [lo, hi])
    w_abs = np.percentile(
        np.abs(np.concatenate([s["w"] for s in snapshots])), CLIP_VORT
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_cp, ax_p = axes[0, 0], axes[0, 1]
    ax_u, ax_w = axes[1, 0], axes[1, 1]

    # pressure coefficient plot (top left)

    if Path(VERIFICATION_PATH).exists():
        ref = np.loadtxt(VERIFICATION_PATH, comments="#")
        ax_cp.plot(ref[:, 0], ref[:, 1], "-", color="C1", label="verification")
    (cp_line,) = ax_cp.plot([], [], "o", color="C0", markersize=4, label="solver")
    ax_cp.set_xlim(*CP_XLIM)
    ax_cp.set_ylim(*CP_YLIM)
    ax_cp.set_xlabel(r"$x$")
    ax_cp.set_ylabel(r"$c_p$")
    ax_cp.invert_yaxis()
    ax_cp.grid(True, alpha=0.3)
    ax_cp.legend(loc="lower right")
    ax_cp.set_title(r"$c_p$")

    # pressure field (top right)

    p_tpc = ax_p.tripcolor(
        triang,
        snapshots[0]["p"],
        shading="gouraud",
        cmap="viridis",
        vmin=p_min,
        vmax=p_max,
    )
    ax_p.fill(body_x, body_y, facecolor="lightgrey", edgecolor="dimgrey", linewidth=1.0)
    ax_p.set_xlim(*FIELD_XLIM)
    ax_p.set_ylim(*FIELD_YLIM)
    ax_p.set_aspect("equal")
    ax_p.set_title(r"$p$")
    fig.colorbar(p_tpc, ax=ax_p)

    # u-velocity field (bottom left)

    u_tpc = ax_u.tripcolor(
        triang,
        snapshots[0]["u"],
        shading="gouraud",
        cmap="viridis",
        vmin=u_min,
        vmax=u_max,
    )
    ax_u.fill(body_x, body_y, facecolor="lightgrey", edgecolor="dimgrey", linewidth=1.0)
    ax_u.set_xlim(*FIELD_XLIM)
    ax_u.set_ylim(*FIELD_YLIM)
    ax_u.set_aspect("equal")
    ax_u.set_title(r"$u$")
    fig.colorbar(u_tpc, ax=ax_u)

    # vorticity field (bottom right)

    w_tpc = ax_w.tripcolor(
        triang,
        snapshots[0]["w"],
        shading="gouraud",
        cmap="RdBu_r",
        vmin=-w_abs,
        vmax=w_abs,
    )
    ax_w.fill(body_x, body_y, facecolor="lightgrey", edgecolor="dimgrey", linewidth=1.0)
    ax_w.set_xlim(*FIELD_XLIM)
    ax_w.set_ylim(*FIELD_YLIM)
    ax_w.set_aspect("equal")
    ax_w.set_title(r"$\omega$")
    fig.colorbar(w_tpc, ax=ax_w)

    suptitle = fig.suptitle("", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96], pad=2.0, w_pad=3.0, h_pad=3.0)

    def update(frame):
        s = snapshots[frame]
        idx = np.argsort(s["xb"])
        cp_line.set_data(s["xb"][idx], (s["pb"] / q)[idx])
        p_tpc.set_array(s["p"])
        u_tpc.set_array(s["u"])
        w_tpc.set_array(s["w"])
        suptitle.set_text(s["name"])
        return [cp_line, p_tpc, u_tpc, w_tpc]

    anim = animation.FuncAnimation(
        fig, update, frames=len(snapshots), interval=1000 / FPS, blit=False
    )
    Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)

    def progress(i, n):
        print(f"\rrendering frame {i + 1}/{n}", end="", flush=True)

    anim.save(SAVE_PATH, writer="ffmpeg", fps=FPS, progress_callback=progress)
    plt.close(fig)
    print(f"\nwrote {SAVE_PATH}")


if __name__ == "__main__":
    animate_data()
