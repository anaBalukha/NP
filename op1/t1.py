"""
Task 1 — Single Robot Path Following
=====================================
Mathematical Model: Initial Value Problem (IVP)

State of robot:  [x, y, vx, vy]  (position + velocity in 2-D)

Governing equations (from spec):
    ẋ = v · min(1, vmax / ‖v‖)              [velocity-saturated kinematics]
    v̇ = (1/m)[kp·(T(t)−x) − kd·v + f_bnd]  [PD control + boundary force]

where:
    T(t) = spline.position(t/T)   — time-varying waypoint sliding along path
    f_bnd = −K_BOUND · overshoot · n̂  — soft wall force if robot leaves road
    frep  = 0  (single robot, no collision avoidance needed)

Numerical method: explicit RK4 (4th-order Runge–Kutta), fixed step Δt.
  Error order: O(Δt⁴) per step, O(Δt⁴) global.
  Stable for the chosen parameters.

Image pipeline:
    RGB → Grayscale → Gaussian blur → Otsu threshold (INVERTED for light paths)
    → Zhang–Suen skeletonisation → BFS shortest path → cubic spline
"""

import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline
from PIL import Image
from skimage.morphology import skeletonize
from collections import deque

warnings.filterwarnings("ignore")

# ── Parameters ─────────────────────────────────────────────────────────────────
M            = 1.0    # robot mass [kg]
KP           = 50.0   # proportional (attraction) gain
KD           = 6.0    # damping gain
K_BOUND      = 500.0  # boundary repulsion gain  (soft wall)
VMAX         = 2    # max speed [m/s]
DT           = 0.02   # RK4 time step [s]
T_SIM        = 30.0   # simulation duration [s]
ROBOT_R      = 0.1    # robot radius [m]  (visual + boundary margin)
PIX2M        = 0.05   # pixel → metre scale
BLUR_SIGMA   = 3.0    # Gaussian pre-blur σ
FPS          = 30     # animation frame rate
# ───────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_image(path: str) -> np.ndarray:
    """Load image, convert to RGB numpy array."""
    return np.array(Image.open(path).convert("RGB"))


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])


def otsu_threshold(arr_uint8: np.ndarray) -> int:
    """Classic Otsu method; returns optimal threshold value 0–255."""
    hist, _ = np.histogram(arr_uint8.flatten(), bins=256, range=(0, 256))
    total = arr_uint8.size
    sum_all = float(np.dot(np.arange(256), hist))
    sum_bg, w_bg, best_var, threshold = 0.0, 0, 0.0, 0
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        mu_bg = sum_bg / w_bg
        mu_fg = (sum_all - sum_bg) / w_fg
        var = w_bg * w_fg * (mu_bg - mu_fg) ** 2
        if var > best_var:
            best_var = var
            threshold = t
    return threshold


def segment_path(gray: np.ndarray) -> np.ndarray:
    """
    Return boolean mask where True = road pixels.
    INVERTED threshold: bright pixels (road on dark background) → True.
    Works equally for dark-road images: just swap to `< threshold`.
    """
    smooth = gaussian_filter(gray.astype(float), sigma=BLUR_SIGMA)
    norm = ((smooth - smooth.min()) / (np.ptp(smooth) + 1e-10) * 255).astype(np.uint8)
    thr = otsu_threshold(norm)
    # INVERTED: road is LIGHTER than background
    binary = norm > thr
    print(f"  Otsu threshold = {thr}  (keeping pixels BRIGHTER than {thr})")
    return binary, smooth, norm


def get_skeleton(binary: np.ndarray) -> np.ndarray:
    """Medial-axis thinning using skimage Zhang–Suen implementation."""
    return skeletonize(binary)


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH & BFS PATH EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def build_adjacency(skeleton: np.ndarray) -> dict:
    """Build 8-connected adjacency dict for skeleton pixels."""
    ys, xs = np.where(skeleton)
    node_set = set(zip(ys.tolist(), xs.tolist()))
    graph = {}
    for y, x in node_set:
        nbrs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb = (y + dy, x + dx)
                if nb in node_set:
                    nbrs.append(nb)
        graph[(y, x)] = nbrs
    return graph


def bfs(graph: dict, start: tuple, goal: tuple):
    """BFS shortest path; returns list of (y, x) nodes or None."""
    if start not in graph or goal not in graph:
        return None
    q = deque([(start, [start])])
    visited = {start}
    while q:
        cur, path = q.popleft()
        if cur == goal:
            return path
        for nb in graph.get(cur, []):
            if nb not in visited:
                visited.add(nb)
                q.append((nb, path + [nb]))
    return None


def snap_to_skeleton(skeleton: np.ndarray, click_yx) -> tuple:
    """Return the skeleton pixel nearest to click_yx=(y,x)."""
    ys, xs = np.where(skeleton)
    coords = np.column_stack([ys, xs])          # (N,2)
    dists = np.linalg.norm(coords - np.array(click_yx), axis=1)
    idx = np.argmin(dists)
    return (int(ys[idx]), int(xs[idx]))


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE POINT SELECTION
# ══════════════════════════════════════════════════════════════════════════════

_selected_points = []


def _on_click(event, fig, ax, skeleton):
    if event.inaxes != ax:
        return
    px, py = int(event.xdata), int(event.ydata)
    snapped = snap_to_skeleton(skeleton, (py, px))
    _selected_points.append(snapped)

    color  = "lime" if len(_selected_points) == 1 else "red"
    marker = "o"    if len(_selected_points) == 1 else "s"
    label  = "A"    if len(_selected_points) == 1 else "B"
    ax.scatter(snapped[1], snapped[0], c=color, s=220,
               marker=marker, edgecolors="black", linewidths=2.5,
               label=label, zorder=15)
    ax.legend(fontsize=11)
    fig.canvas.draw()
    if len(_selected_points) == 2:
        plt.close(fig)


def select_ab(rgb: np.ndarray, skeleton: np.ndarray):
    """Show image + skeleton overlay; user clicks A then B."""
    global _selected_points
    _selected_points = []

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(rgb)
    overlay = np.zeros((*skeleton.shape, 4), dtype=float)
    overlay[skeleton] = [1.0, 0.2, 0.2, 0.55]
    ax.imshow(overlay)
    ax.set_title("Click  Point A  (start)  then  Point B  (end)",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.canvas.mpl_connect("button_press_event",
                           lambda e: _on_click(e, fig, ax, skeleton))
    plt.tight_layout()
    plt.show()

    if len(_selected_points) != 2:
        raise RuntimeError("Exactly 2 points required.")
    return _selected_points[0], _selected_points[1]


# ══════════════════════════════════════════════════════════════════════════════
#  CUBIC-SPLINE PATH REPRESENTATION
# ══════════════════════════════════════════════════════════════════════════════

class PathSpline:
    """
    Arc-length parametrised cubic spline of the extracted centerline.
    Coordinates stored in metres.  s ∈ [0, 1].
    """

    def __init__(self, xy_pixels, path_width_pix: float, scale: float = PIX2M):
        self.scale      = scale
        self.path_width = path_width_pix * scale   # metres

        pts = np.array(xy_pixels, dtype=float) * scale  # → metres (x, y)

        # Arc-length parameter
        diffs   = np.diff(pts, axis=0)
        seglens = np.linalg.norm(diffs, axis=1)
        s_raw   = np.concatenate([[0.0], np.cumsum(seglens)])
        self.total_length = s_raw[-1]

        # Normalise s ∈ [0, 1] and remove duplicates
        s_norm = s_raw / self.total_length
        _, uniq = np.unique(s_norm, return_index=True)
        s_norm, pts = s_norm[uniq], pts[uniq]

        self._cx  = CubicSpline(s_norm, pts[:, 0], bc_type="natural")
        self._cy  = CubicSpline(s_norm, pts[:, 1], bc_type="natural")
        self._dcx = self._cx.derivative()
        self._dcy = self._cy.derivative()

    # ── geometric queries ──────────────────────────────────────────────────

    def position(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, 1.0))
        return np.array([self._cx(s), self._cy(s)])

    def tangent(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, 1.0))
        t = np.array([self._dcx(s), self._dcy(s)])
        return t / (np.linalg.norm(t) + 1e-12)

    def normal(self, s: float) -> np.ndarray:
        t = self.tangent(s)
        return np.array([-t[1], t[0]])

    def closest_s(self, point: np.ndarray, n: int = 500) -> float:
        s_arr = np.linspace(0.0, 1.0, n)
        curve = np.column_stack([self._cx(s_arr), self._cy(s_arr)])
        return float(s_arr[np.argmin(np.linalg.norm(curve - point, axis=1))])

    def boundary_force(self, pos: np.ndarray) -> np.ndarray:
        """
        Soft wall: zero inside road, proportional restoring force outside.
        max_lat = path_width/2 − ROBOT_R  ensures the robot body stays inside.
        """
        s      = self.closest_s(pos)
        center = self.position(s)
        n_vec  = self.normal(s)
        lat    = float(np.dot(pos - center, n_vec))   # signed lateral offset
        max_lat = self.path_width / 2.0 - ROBOT_R
        if abs(lat) > max_lat:
            overshoot = abs(lat) - max_lat
            return -np.sign(lat) * n_vec * K_BOUND * overshoot
        return np.zeros(2)


# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMICS  (IVP formulation, single robot)
# ══════════════════════════════════════════════════════════════════════════════

def _sat(v: np.ndarray) -> np.ndarray:
    """Apply velocity saturation:  v_sat = v · min(1, vmax / ‖v‖)."""
    n = np.linalg.norm(v)
    return v * min(1.0, VMAX / n) if n > 1e-12 else v


def _rhs(t: float, state: np.ndarray, target_fn, spline: PathSpline) -> np.ndarray:
    """
    Right-hand side of the IVP:
        ẋ = sat(v)
        v̇ = (1/m)[kp·(T(t)−x) − kd·v + f_bnd(x)]
    """
    x, v   = state[:2], state[2:]
    T_t    = target_fn(t)
    f_attr = KP * (T_t - x)
    f_damp = -KD * v
    f_bnd  = spline.boundary_force(x)
    return np.concatenate([_sat(v), (f_attr + f_damp + f_bnd) / M])


def _rk4_step(t, state, dt, rhs_fn, *args) -> np.ndarray:
    """Single RK4 step.  Butcher tableau: classical 4-stage, order 4."""
    k1 = rhs_fn(t,        state,            *args)
    k2 = rhs_fn(t + dt/2, state + dt/2*k1, *args)
    k3 = rhs_fn(t + dt/2, state + dt/2*k2, *args)
    k4 = rhs_fn(t + dt,   state + dt  *k3, *args)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate(spline: PathSpline, T: float = T_SIM, dt: float = DT):
    """
    Integrate IVP from t=0 to t=T.
    Initial conditions: x=A, v=0.
    Target T(t) = spline.position(t/T)  — slides from A to B over [0,T].
    Returns (t_array, state_history).
    """
    pos_A  = spline.position(0.0)
    pos_B  = spline.position(1.0)
    state  = np.array([pos_A[0], pos_A[1], 0.0, 0.0])

    def target_fn(t):
        return spline.position(min(t / T, 1.0))

    t_arr   = np.arange(0.0, T + dt, dt)
    history = np.zeros((len(t_arr), 4))
    history[0] = state

    for i, t in enumerate(t_arr[:-1]):
        state = _rk4_step(t, state, dt, _rhs, target_fn, spline)
        # Hard velocity cap (saturation guard after integration)
        n = np.linalg.norm(state[2:])
        if n > VMAX:
            state[2:] *= VMAX / n
        history[i + 1] = state

    err = np.linalg.norm(history[-1, :2] - pos_B)
    print(f"  Start  : {pos_A}")
    print(f"  Goal   : {pos_B}")
    print(f"  Final position error : {err:.4f} m")
    return t_arr, history


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _draw_road(ax, spline: PathSpline, n: int = 300):
    s_arr = np.linspace(0, 1, n)
    cl  = np.array([spline.position(s) for s in s_arr]) / PIX2M
    lft = np.array([spline.position(s) + spline.path_width/2 * spline.normal(s)
                    for s in s_arr]) / PIX2M
    rgt = np.array([spline.position(s) - spline.path_width/2 * spline.normal(s)
                    for s in s_arr]) / PIX2M
    ax.plot(lft[:, 0], lft[:, 1], "b-",  lw=2.0, alpha=0.85, label="Road boundary")
    ax.plot(rgt[:, 0], rgt[:, 1], "b-",  lw=2.0, alpha=0.85)
    ax.plot(cl[:, 0],  cl[:, 1],  "w--", lw=1.0, alpha=0.50, label="Centreline")
    return cl


def plot_pipeline(rgb, gray, smooth, binary, skeleton, pts_pix,
                  spline, ptA, ptB, prefix="task1"):
    """8-panel extraction pipeline figure."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))

    def off(a):
        a.axis("off")

    axes[0, 0].imshow(rgb);                       axes[0, 0].set_title("1. Original RGB",           fontweight="bold"); off(axes[0, 0])
    axes[0, 1].imshow(gray,   cmap="gray");        axes[0, 1].set_title("2. Grayscale",               fontweight="bold"); off(axes[0, 1])
    axes[0, 2].imshow(smooth, cmap="gray");        axes[0, 2].set_title(f"3. Gaussian blur  σ={BLUR_SIGMA}", fontweight="bold"); off(axes[0, 2])
    axes[0, 3].imshow(binary, cmap="gray");        axes[0, 3].set_title("4. Otsu  (inverted — bright=road)", fontweight="bold"); off(axes[0, 3])
    axes[1, 0].imshow(skeleton, cmap="gray");      axes[1, 0].set_title("5. Skeleton  (Zhang–Suen)",  fontweight="bold"); off(axes[1, 0])

    # Points A & B on original + skeleton overlay
    ax = axes[1, 1];   ax.imshow(rgb)
    ov = np.zeros((*skeleton.shape, 4), float);  ov[skeleton] = [1, 0.2, 0.2, 0.5]
    ax.imshow(ov)
    ax.scatter(ptA[1], ptA[0], c="lime", s=220, marker="o", edgecolors="k", lw=2.5, label="A", zorder=10)
    ax.scatter(ptB[1], ptB[0], c="red",  s=220, marker="s", edgecolors="k", lw=2.5, label="B", zorder=10)
    ax.set_title("6. Selected Points A & B", fontweight="bold"); ax.legend(); off(ax)

    # BFS pixel path
    ax = axes[1, 2];   ax.imshow(rgb)
    px = [p[0] for p in pts_pix];  py = [p[1] for p in pts_pix]
    ax.plot(px, py, "r-", lw=2)
    ax.scatter(px[0], py[0],  c="lime", s=150, edgecolors="k", lw=2, label="A", zorder=5)
    ax.scatter(px[-1], py[-1], c="blue", s=150, marker="s", edgecolors="k", lw=2, label="B", zorder=5)
    ax.set_title("7. BFS Path A → B", fontweight="bold"); ax.legend(); off(ax)

    # Cubic spline (in pixel coords)
    ax = axes[1, 3]
    s_arr   = np.linspace(0, 1, 500)
    curve   = np.array([spline.position(s) for s in s_arr]) / PIX2M
    ax.plot(px, py, "ro", ms=4, alpha=0.5, label="Raw points")
    ax.plot(curve[:, 0], curve[:, 1], "b-", lw=2, label="Cubic spline")
    ax.set_title("8. Arc-length Parametrised Spline", fontweight="bold")
    ax.legend();  ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [pixels]"); ax.set_ylabel("y [pixels]")
    ax.invert_yaxis()

    plt.tight_layout()
    fname = f"{prefix}_pipeline.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def animate_task1(rgb, spline, t_arr, history, out="task1_robot.gif"):
    """Animate robot following the spline on the real map background."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(rgb)

    cl = _draw_road(ax, spline)
    ax.scatter(*cl[0],  c="lime", s=220, edgecolors="k", lw=2.5, label="A", zorder=10)
    ax.scatter(*cl[-1], c="red",  s=220, marker="s", edgecolors="k", lw=2.5, label="B", zorder=10)

    # Convert history (metres) → pixels for display
    pos_pix = history[:, :2] / PIX2M          # (T, 2)  col, row

    trace, = ax.plot([], [], color="yellow", lw=2, alpha=0.9, label="Trace")
    robot   = Circle((0, 0), ROBOT_R / PIX2M, color="cyan", alpha=0.85,
                     edgecolor="white", lw=2, zorder=15)
    ax.add_patch(robot)

    vel_arr = np.linalg.norm(history[:, 2:], axis=1)

    txt_time = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=12,
                       va="top", bbox=dict(boxstyle="round", fc="wheat", alpha=0.9))
    txt_vel  = ax.text(0.02, 0.90, "", transform=ax.transAxes, fontsize=11,
                       va="top", bbox=dict(boxstyle="round", fc="lightblue", alpha=0.9))

    ax.set_title("Task 1 — Single Robot Path Following  (IVP + RK4)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right"); ax.axis("off")

    step   = max(1, len(t_arr) // 150)
    frames = range(0, len(t_arr), step)

    def update(f):
        robot.center = (pos_pix[f, 0], pos_pix[f, 1])
        trace.set_data(pos_pix[:f+1, 0], pos_pix[:f+1, 1])
        txt_time.set_text(f"t = {t_arr[f]:.2f} s")
        txt_vel.set_text(f"‖v‖ = {vel_arr[f]:.3f} m/s")
        return robot, trace, txt_time, txt_vel

    anim = FuncAnimation(fig, update, frames=frames, interval=40, blit=False)
    anim.save(out, writer=PillowWriter(fps=FPS))
    plt.close()
    print(f"  Saved: {out}")


def plot_results(spline, t_arr, history, prefix="task1"):
    """Plot position error, speed, lateral deviation over time."""
    pos   = history[:, :2]
    vel   = np.linalg.norm(history[:, 2:], axis=1)
    goal  = spline.position(1.0)
    err   = np.linalg.norm(pos - goal, axis=1)

    # Lateral deviation
    lat_dev = np.array([abs(spline.boundary_force(pos[i])[0])  # proxy; compute properly
                        for i in range(0, len(pos), max(1, len(pos)//200))])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(t_arr, err, "b-", lw=1.5)
    axes[0].set(xlabel="Time [s]", ylabel="Distance to goal [m]",
                title="Position Error over Time"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_arr, vel, "g-", lw=1.5)
    axes[1].axhline(VMAX, color="r", ls="--", label=f"vmax={VMAX}")
    axes[1].set(xlabel="Time [s]", ylabel="Speed [m/s]", title="Robot Speed")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(np.linspace(0, t_arr[-1], len(lat_dev)), lat_dev, "m-", lw=1.5)
    axes[2].set(xlabel="Time [s]", ylabel="|Boundary force| [N]",
                title="Boundary Force (non-zero = violation)"); axes[2].grid(True, alpha=0.3)

    plt.suptitle("Task 1 — Simulation Results", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"{prefix}_results.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  TASK 1 — Single Robot Path Following (IVP + RK4)")
    print("=" * 60)

    map_path  = "Screenshot 2026-02-17 190607.png"
    width_pix = 20

    # ── Step 1: Image pipeline ────────────────────────────────────────────
    print("\n[1/6] Image preprocessing …")
    rgb    = load_image(map_path)
    gray   = to_grayscale(rgb)
    binary, smooth, norm8 = segment_path(gray)
    skeleton = get_skeleton(binary)
    print(f"  Image size: {rgb.shape[1]}×{rgb.shape[0]} px")
    print(f"  Skeleton pixels: {skeleton.sum()}")

    # ── Step 2: Select A & B ─────────────────────────────────────────────
    print("\n[2/6] Select start (A) and end (B) on the map …")
    ptA, ptB = select_ab(rgb, skeleton)
    print(f"  A = {ptA}  (row, col)")
    print(f"  B = {ptB}  (row, col)")

    # ── Step 3: BFS path & spline ─────────────────────────────────────────
    print("\n[3/6] Extracting centreline path …")
    graph = build_adjacency(skeleton)
    raw_path = bfs(graph, ptA, ptB)
    if raw_path is None:
        print("ERROR: No path found between A and B.  "
              "Try different points or check the threshold polarity.")
        sys.exit(1)

    # Convert (y, x) pixels → (x, y) pixels, decimate
    pts_xy = [[p[1], p[0]] for p in raw_path]
    if len(pts_xy) > 120:
        idx = list(range(0, len(pts_xy), max(1, len(pts_xy) // 120)))
        if idx[-1] != len(pts_xy) - 1:
            idx.append(len(pts_xy) - 1)
        pts_xy = [pts_xy[i] for i in idx]

    spline = PathSpline(pts_xy, width_pix)
    print(f"  Path length  : {spline.total_length:.3f} m  ({len(pts_xy)} control points)")
    print(f"  Road width   : {spline.path_width:.3f} m  ({width_pix:.0f} px)")

    # ── Step 4: Visualise pipeline ────────────────────────────────────────
    print("\n[4/6] Saving pipeline figure …")
    plot_pipeline(rgb, gray, smooth, binary, skeleton,
                  pts_xy, spline, ptA, ptB, prefix="task1")

    # ── Step 5: Simulate ─────────────────────────────────────────────────
    print(f"\n[5/6] Simulating IVP  (T={T_SIM} s, Δt={DT} s, RK4) …")
    t_arr, history = simulate(spline)
    plot_results(spline, t_arr, history)

    # ── Step 6: Animate ───────────────────────────────────────────────────
    print("\n[6/6] Rendering animation …")
    animate_task1(rgb, spline, t_arr, history)

    print("\n✓  Task 1 complete!")
    print("   Outputs: task1_pipeline.png   task1_results.png   task1_robot.gif")


if __name__ == "__main__":
    main()