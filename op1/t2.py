import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline
from PIL import Image
from skimage.morphology import skeletonize
from collections import deque
import random

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ── Parameters ─────────────────────────────────────────────────────────────────
M          = 1.0    # robot mass [kg]
KP         = 50.0   # proportional gain  (higher → faster response to waypoint)
KD         = 8.0    # damping gain        (higher → less overshoot)
KREP       = 15.0   # repulsion gain      (lower → less sideways push into walls)
RSAFE      = 0.6    # safety radius [m]   (smaller → repulsion only when truly close)
K_BOUND    = 3000.0 # boundary wall gain  (very stiff — must beat repulsion peaks)
VMAX       = 1.0    # max speed [m/s]
DT         = 0.02   # RK4 time step [s]
T_ONE_WAY  = 35.0   # time per leg — longer gives robots time to complete the trip
T_SIM      = 2 * T_ONE_WAY   # full round-trip
ROBOT_R    = 0.1    # robot radius [m]
PIX2M      = 0.05   # pixel → metre
BLUR_SIGMA = 3.0
FPS        = 30
N_PER_DIR  = 4      # robots per direction (total = 2×N_PER_DIR)
SPAWN_JITTER = 0.15 # tight spawn — keep robots on path from the start
# ───────────────────────────────────────────────────────────────────────────────

def load_image(path):
    return np.array(Image.open(path).convert("RGB"))

def to_grayscale(rgb):
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

def otsu_threshold(arr8):
    hist, _ = np.histogram(arr8.flatten(), bins=256, range=(0, 256))
    total = arr8.size
    s_all = float(np.dot(np.arange(256), hist))
    s_bg = w_bg = best = thr = 0
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0: continue
        w_fg = total - w_bg
        if w_fg == 0: break
        s_bg += t * hist[t]
        v = w_bg * w_fg * ((s_bg / w_bg) - (s_all - s_bg) / w_fg) ** 2
        if v > best:
            best = v; thr = t
    return thr

def segment_path(gray):
    """Gaussian blur → Otsu threshold → INVERTED (bright = road)."""
    smooth = gaussian_filter(gray, sigma=BLUR_SIGMA)
    arr8   = ((smooth - smooth.min()) / (np.ptp(smooth) + 1e-10) * 255).astype(np.uint8)
    thr    = otsu_threshold(arr8)
    binary = arr8 > thr
    print(f"  Otsu threshold={thr}  (keeping pixels > {thr}  =  bright = road)")
    return binary, smooth

def get_skeleton(binary):
    return skeletonize(binary)

def build_adjacency(skeleton):
    ys, xs = np.where(skeleton)
    node_set = set(zip(ys.tolist(), xs.tolist()))
    graph = {}
    for y, x in node_set:
        nbrs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0: continue
                nb = (y + dy, x + dx)
                if nb in node_set: nbrs.append(nb)
        graph[(y, x)] = nbrs
    return graph

def bfs(graph, start, goal):
    if start not in graph or goal not in graph: return None
    q = deque([(start, [start])])
    visited = {start}
    while q:
        cur, path = q.popleft()
        if cur == goal: return path
        for nb in graph.get(cur, []):
            if nb not in visited:
                visited.add(nb); q.append((nb, path + [nb]))
    return None

def snap_to_skeleton(skeleton, yx):
    ys, xs = np.where(skeleton)
    coords = np.column_stack([ys, xs])
    idx = np.argmin(np.linalg.norm(coords - np.array(yx), axis=1))
    return (int(ys[idx]), int(xs[idx]))

_sel = []

def _on_click(event, fig, ax, skeleton):
    if event.inaxes != ax: return
    snapped = snap_to_skeleton(skeleton, (int(event.ydata), int(event.xdata)))
    _sel.append(snapped)
    color, marker, label = ("lime", "o", "A") if len(_sel) == 1 else ("red", "s", "B")
    ax.scatter(snapped[1], snapped[0], c=color, s=220,
               marker=marker, edgecolors="k", lw=2.5, label=label, zorder=15)
    ax.legend(fontsize=11); fig.canvas.draw()
    if len(_sel) == 2: plt.close(fig)

def select_ab(rgb, skeleton):
    global _sel; _sel = []
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(rgb)
    ax.set_title("Click  Point A  (start)  then  Point B  (end)",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.canvas.mpl_connect("button_press_event", lambda e: _on_click(e, fig, ax, skeleton))
    plt.tight_layout(); plt.show()
    if len(_sel) != 2: raise RuntimeError("Need exactly 2 points.")
    return _sel[0], _sel[1]

class PathSpline:

    def __init__(self, xy_pix, path_width_pix, scale=PIX2M):
        self.scale = scale
        self.path_width = path_width_pix * scale
        pts = np.array(xy_pix, float) * scale
        diffs   = np.diff(pts, axis=0)
        seglens = np.linalg.norm(diffs, axis=1)
        s_raw   = np.concatenate([[0.], np.cumsum(seglens)])
        self.total_length = s_raw[-1]
        s_norm = s_raw / self.total_length
        _, uniq = np.unique(s_norm, return_index=True)
        s_norm, pts = s_norm[uniq], pts[uniq]
        self._cx  = CubicSpline(s_norm, pts[:, 0], bc_type="natural")
        self._cy  = CubicSpline(s_norm, pts[:, 1], bc_type="natural")
        self._dcx = self._cx.derivative()
        self._dcy = self._cy.derivative()

    def position(self, s):
        s = float(np.clip(s, 0., 1.))
        return np.array([self._cx(s), self._cy(s)])

    def tangent(self, s):
        s = float(np.clip(s, 0., 1.))
        t = np.array([self._dcx(s), self._dcy(s)])
        return t / (np.linalg.norm(t) + 1e-12)

    def normal(self, s):
        t = self.tangent(s)
        return np.array([-t[1], t[0]])

    def closest_s(self, pt, n_coarse=300, n_fine=200):
        pt = np.asarray(pt)
        # Coarse pass over full spline
        s_coarse = np.linspace(0., 1., n_coarse)
        curve    = np.column_stack([self._cx(s_coarse), self._cy(s_coarse)])
        best_i   = int(np.argmin(np.linalg.norm(curve - pt, axis=1)))
        # Fine pass in a tight window around the coarse best
        s_lo = s_coarse[max(0, best_i - 3)]
        s_hi = s_coarse[min(n_coarse - 1, best_i + 3)]
        s_fine = np.linspace(s_lo, s_hi, n_fine)
        curve2 = np.column_stack([self._cx(s_fine), self._cy(s_fine)])
        best_j = int(np.argmin(np.linalg.norm(curve2 - pt, axis=1)))
        return float(s_fine[best_j])

    def boundary_force(self, pos):
        s     = self.closest_s(pos)
        n_vec = self.normal(s)
        lat   = float(np.dot(pos - self.position(s), n_vec))
        # Safe corridor is ± (path_width/2 − ROBOT_R)
        mlat  = self.path_width / 2. - ROBOT_R
        if abs(lat) > mlat:
            overshoot = abs(lat) - mlat
            # Linear + quadratic spring (very stiff wall)
            mag = K_BOUND * overshoot + K_BOUND * 5.0 * overshoot ** 2
            return -np.sign(lat) * n_vec * mag
        return np.zeros(2)

    def lateral_error(self, pos):
        s     = self.closest_s(pos)
        lat   = float(np.dot(pos - self.position(s), self.normal(s)))
        return lat

def _sat(v):
    n = np.linalg.norm(v)
    return v * (VMAX / n) if n > VMAX else v


def swarm_rhs(t, states, target_fns, spline):
    N = len(states) // 4
    flat = states.reshape(N, 4)
    pos  = flat[:, :2]   # (N,2)
    vel  = flat[:, 2:]   # (N,2)
    dstates = np.zeros_like(flat)

    for i in range(N):
        xi, vi = pos[i], vel[i]

        # Kinematics
        dstates[i, :2] = _sat(vi)

        # Target attraction
        Ti = target_fns[i](t)
        f  = KP * (Ti - xi) - KD * vi

        # Pairwise repulsion (frep only inside RSAFE)
        for j in range(N):
            if j == i: continue
            diff = xi - pos[j]
            d    = np.linalg.norm(diff) + 1e-10
            if d < RSAFE:
                f += KREP * diff / d ** 3

        # Soft-wall boundary
        f += spline.boundary_force(xi)

        dstates[i, 2:] = f / M

    return dstates.flatten()

def _rk4(t, s, dt, rhs, *args):
    k1 = rhs(t,        s,             *args)
    k2 = rhs(t + dt/2, s + dt/2 * k1, *args)
    k3 = rhs(t + dt/2, s + dt/2 * k2, *args)
    k4 = rhs(t + dt,   s + dt   * k3, *args)
    return s + (dt / 6.) * (k1 + 2*k2 + 2*k3 + k4)


def _rhs_fixed(states, targets_list, spline, N):
    flat = states.reshape(N, 4)
    pos  = flat[:, :2]
    vel  = flat[:, 2:]
    dstates = np.zeros_like(flat)
    for i in range(N):
        xi, vi = pos[i], vel[i]
        dstates[i, :2] = _sat(vi)
        Ti = targets_list[i]
        f  = KP * (Ti - xi) - KD * vi
        for j in range(N):
            if j == i: continue
            diff = xi - pos[j]
            d    = np.linalg.norm(diff) + 1e-10
            if d < RSAFE:
                f += KREP * diff / d ** 3
        f += spline.boundary_force(xi)
        dstates[i, 2:] = f / M
    return dstates.flatten()


def _rk4_fixed_targets(t, s, dt, targets_list, spline, N):
    def rhs(state):
        return _rhs_fixed(state, targets_list, spline, N)
    k1 = rhs(s)
    k2 = rhs(s + dt/2 * k1)
    k3 = rhs(s + dt/2 * k2)
    k4 = rhs(s + dt   * k3)
    return s + (dt / 6.) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_swarm(spline, n_per_dir=N_PER_DIR, dt=DT, T=T_SIM, T1=T_ONE_WAY):

    N     = 2 * n_per_dir
    posA  = spline.position(0.)
    posB  = spline.position(1.)

    state0 = np.zeros(4 * N)
    for i in range(n_per_dir):
        idx = i * 4
        state0[idx:idx+2] = posA + np.random.randn(2) * SPAWN_JITTER
    for i in range(n_per_dir):
        idx = (n_per_dir + i) * 4
        state0[idx:idx+2] = posB + np.random.randn(2) * SPAWN_JITTER

    progress = np.zeros(N)
    leg      = np.zeros(N, dtype=int)
    group = np.array([0]*n_per_dir + [1]*n_per_dir)

    def get_target(i):
        p = progress[i]
        g = group[i]
        l = leg[i]
        if (g == 0 and l == 0) or (g == 1 and l == 1):
            s = p            # travelling toward B
        else:
            s = 1.0 - p      # travelling toward A
        return spline.position(np.clip(s, 0., 1.))

    ds_per_metre = 1.0 / spline.total_length

    t_arr   = np.arange(0., T + dt, dt)
    history = np.zeros((len(t_arr), 4 * N))
    state   = state0.copy()
    history[0] = state

    for i, t in enumerate(t_arr[:-1]):
        target_fns = [get_target(j) for j in range(N)]

        state_new = _rk4_fixed_targets(t, state, dt, target_fns, spline, N)

        for j in range(N):
            v = state_new[j*4+2 : j*4+4]
            n_ = np.linalg.norm(v)
            if n_ > VMAX:
                state_new[j*4+2 : j*4+4] *= VMAX / n_

        for j in range(N):
            speed = np.linalg.norm(state_new[j*4+2 : j*4+4])
            progress[j] += speed * dt * ds_per_metre
            if progress[j] >= 1.0:
                progress[j] = 0.0   # start return leg
                leg[j] = 1 - leg[j]

        state = state_new
        history[i+1] = state

    completed = [leg[j] for j in range(N)]
    print(f"  Simulated {N} robots for {T:.0f} s  ({len(t_arr)} steps)")
    print(f"  Legs completed per robot: {completed}  (1=done round trip, 0=still on first leg)")
    return t_arr, history, N


# ══════════════════════════════════════════════════════════════════════════════
#  COLLISION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(history, N, spline):
    min_dists = []
    violations = []
    for state in history:
        pos = state.reshape(N, 4)[:, :2]
        dists = []
        for i in range(N):
            for j in range(i + 1, N):
                dists.append(np.linalg.norm(pos[i] - pos[j]))
        min_dists.append(min(dists) if dists else 0.)
        v = sum(1 for i in range(N)
                if abs(spline.lateral_error(pos[i])) > spline.path_width / 2)
        violations.append(v)
    return np.array(min_dists), np.array(violations)


def plot_metrics(t_arr, min_dists, violations, n_per_dir, out="task2_metrics.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Min distance vs time
    ax = axes[0]
    ax.plot(t_arr, min_dists, "b-", lw=1.5, label="Min inter-robot dist")
    ax.axhline(RSAFE,        color="orange", ls="--", lw=2, label=f"Rsafe = {RSAFE} m")
    ax.axhline(2 * ROBOT_R,  color="red",    ls="--", lw=2, label=f"Collision = {2*ROBOT_R} m")
    ax.fill_between(t_arr, 0, np.minimum(min_dists, RSAFE), alpha=0.15, color="orange")
    # Mark half-way (direction reversal)
    ax.axvline(T_ONE_WAY, color="purple", ls=":", lw=1.5, label="Direction reversal")
    ax.set(xlabel="Time [s]", ylabel="Distance [m]", title="Min Inter-Robot Distance")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. Distance histogram
    ax = axes[1]
    ax.hist(min_dists, bins=50, color="steelblue", edgecolor="k", alpha=0.8)
    ax.axvline(RSAFE,       color="orange", ls="--", lw=2, label=f"Rsafe")
    ax.axvline(2 * ROBOT_R, color="red",    ls="--", lw=2, label="Collision")
    ax.set(xlabel="Min distance [m]", ylabel="Frequency", title="Min Distance Histogram")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # 3. Boundary violations
    ax = axes[2]
    ax.plot(t_arr, violations, "r-", lw=1.5)
    ax.fill_between(t_arr, 0, violations, alpha=0.25, color="red")
    ax.set(xlabel="Time [s]", ylabel="# robots outside road",
           title="Road Boundary Violations")
    ax.grid(True, alpha=0.3)

    n_collisions = int((min_dists < 2 * ROBOT_R).sum())
    t_under_Rsafe = float((min_dists < RSAFE).sum()) * DT
    fig.suptitle(
        f"Task 2 Metrics — {2*n_per_dir} robots  |  "
        f"Hard collisions (d<2R): {n_collisions} steps  |  "
        f"Time under Rsafe: {t_under_Rsafe:.1f} s",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


COLORS_A = ["#CDE2F9", "#AAC4E8", "#A4CCEE", "#B3CFEE"]
COLORS_B = ["#D98B8B", "#EF9696", "#ECAC87", "#EF9090"] 


def _draw_road(ax, spline, n=200):
    s_arr = np.linspace(0, 1, n)
    cl  = np.column_stack([spline.position(s) for s in s_arr]).T / PIX2M
    lft = np.column_stack([spline.position(s) + spline.path_width/2 * spline.normal(s)
                           for s in s_arr]).T / PIX2M
    rgt = np.column_stack([spline.position(s) - spline.path_width/2 * spline.normal(s)
                           for s in s_arr]).T / PIX2M
    ax.plot(lft[:, 0], lft[:, 1], "cyan",    lw=2.5, alpha=0.8, label="Road boundary")
    ax.plot(rgt[:, 0], rgt[:, 1], "cyan",    lw=2.5, alpha=0.8)
    ax.plot(cl[:, 0],  cl[:, 1],  "yellow",  lw=1.5, alpha=0.5, ls="--", label="Centreline")
    return cl

def animate_swarm(rgb, spline, t_arr, history, N, n_per_dir, out="task2_swarm.gif"):
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(rgb)
    cl = _draw_road(ax, spline)

    ax.scatter(*cl[0],  c="lime", s=250, edgecolors="k", lw=2.5, label="A",    zorder=10)
    ax.scatter(*cl[-1], c="red",  s=250, marker="s", edgecolors="k", lw=2.5, label="B", zorder=10)

    robot_r_pix = ROBOT_R / PIX2M
    circles = []
    traces  = []
    for i in range(N):
        c = COLORS_A[i % len(COLORS_A)] if i < n_per_dir else COLORS_B[i % len(COLORS_B)]
        lbl = "→B" if (i == 0) else ("→A" if i == n_per_dir else None)
        circ = Circle((0, 0), robot_r_pix, color=c, alpha=0.85,
                      edgecolor="white", lw=1.5, zorder=12,
                      label=(("Group A" if lbl == "→B" else "Group B") if lbl else None))
        ax.add_patch(circ)
        circles.append(circ)
        line, = ax.plot([], [], color=c, lw=1.2, alpha=0.5)
        traces.append(line)

    txt_time = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=12, va="top",
                       bbox=dict(boxstyle="round", fc="wheat", alpha=0.9))
    txt_info = ax.text(0.02, 0.89, "", transform=ax.transAxes, fontsize=10, va="top",
                       bbox=dict(boxstyle="round", fc="lightblue", alpha=0.9))

    ax.set_title(
        f"Task 2 — Bidirectional Swarm  ({N} robots)  Blue: A→B→A   Red: B→A→B\n"
        f"Road width = {spline.path_width:.2f} m    Rsafe = {RSAFE} m",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right"); ax.axis("off")

    step   = max(1, len(t_arr) // 200)
    frames = range(0, len(t_arr), step)

    def update(f):
        s = history[f]
        pos_pix = s.reshape(N, 4)[:, :2] / PIX2M   # pixels
        for i in range(N):
            circles[i].center = (pos_pix[i, 0], pos_pix[i, 1])
            hist_pix = history[:f+1].reshape(-1, N, 4)[:, i, :2] / PIX2M
            traces[i].set_data(hist_pix[:, 0], hist_pix[:, 1])

        # Min inter-robot distance
        dists = [np.linalg.norm(s.reshape(N, 4)[i, :2] - s.reshape(N, 4)[j, :2])
                 for i in range(N) for j in range(i+1, N)]
        md = min(dists)
        elapsed = t_arr[f]; half = T_ONE_WAY
        phase = "Leg 1: →" if elapsed < half else "Leg 2: ←"
        txt_time.set_text(f"t = {t_arr[f]:.2f} s  |  {phase}")
        color_md = "red" if md < 2 * ROBOT_R else ("orange" if md < RSAFE else "green")
        txt_info.set_text(f"Min dist = {md:.3f} m")
        txt_info.get_bbox_patch().set_facecolor(
            {"red": "#ffaaaa", "orange": "#ffe8aa", "green": "#aaffaa"}[color_md])
        return circles + traces + [txt_time, txt_info]

    anim = FuncAnimation(fig, update, frames=frames, interval=40, blit=False)
    anim.save(out, writer=PillowWriter(fps=FPS))
    plt.close()
    print(f"  Saved: {out}")

def plot_pipeline(rgb, binary, skeleton, pts_pix, spline, ptA, ptB, prefix="task2"):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(rgb);                      axes[0].set_title("Original RGB")
    axes[1].imshow(binary, cmap="gray");      axes[1].set_title("Otsu (inverted)")
    axes[2].imshow(skeleton, cmap="gray");    axes[2].set_title("Skeleton")
    ax = axes[3]; ax.imshow(rgb)
    ov = np.zeros((*skeleton.shape, 4), float); ov[skeleton] = [1, 0.2, 0.2, 0.4]
    ax.imshow(ov)
    px = [p[0] for p in pts_pix]; py = [p[1] for p in pts_pix]
    ax.plot(px, py, "r-", lw=2)
    ax.scatter(ptA[1], ptA[0], c="lime", s=200, edgecolors="k", lw=2, label="A", zorder=10)
    ax.scatter(ptB[1], ptB[0], c="red",  s=200, marker="s", edgecolors="k", lw=2, label="B", zorder=10)
    ax.set_title("BFS Path A→B"); ax.legend()
    for a in axes: a.axis("off")
    plt.tight_layout()
    fname = f"{prefix}_pipeline.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")

def main():
    print("=" * 65)
    print("  TASK 2 — Bidirectional Swarm Navigation  (IVP + RK4)")
    print("=" * 65)

    map_path   = 'Screenshot 2026-02-17 222321.png'
    width_pix  = float(input("Road width in pixels (default: 50)  : ").strip() or 50)
    n_per_dir  = int(input(f"Robots per direction (default: {N_PER_DIR}) : ").strip() or N_PER_DIR)

    # ── Step 1: Image pipeline ────────────────────────────────────────────
    print("\n[1/6] Image preprocessing …")
    rgb      = load_image(map_path)
    gray     = to_grayscale(rgb)
    binary, smooth = segment_path(gray)
    skeleton = get_skeleton(binary)
    print(f"  Image size   : {rgb.shape[1]}×{rgb.shape[0]} px")
    print(f"  Skeleton px  : {skeleton.sum()}")

    # ── Step 2: Select A & B ─────────────────────────────────────────────
    print("\n[2/6] Select start (A) and end (B) on the map …")
    ptA, ptB = select_ab(rgb, skeleton)
    print(f"  A = {ptA}  B = {ptB}  (row, col)")

    # ── Step 3: BFS + spline ─────────────────────────────────────────────
    print("\n[3/6] Extracting centreline …")
    graph    = build_adjacency(skeleton)
    raw_path = bfs(graph, ptA, ptB)
    if raw_path is None:
        print("ERROR: No path found between A and B.")
        sys.exit(1)

    pts_xy = [[p[1], p[0]] for p in raw_path]
    if len(pts_xy) > 120:
        idx = list(range(0, len(pts_xy), max(1, len(pts_xy) // 120)))
        if idx[-1] != len(pts_xy) - 1: idx.append(len(pts_xy) - 1)
        pts_xy = [pts_xy[i] for i in idx]

    spline = PathSpline(pts_xy, width_pix)
    print(f"  Path length : {spline.total_length:.3f} m   Road width : {spline.path_width:.3f} m")
    print(f"  Control pts : {len(pts_xy)}")

    # ── Step 4: Pipeline figure ───────────────────────────────────────────
    print("\n[4/6] Saving pipeline figure …")
    plot_pipeline(rgb, binary, skeleton, pts_xy, spline, ptA, ptB, prefix="task2")

    # ── Step 5: Simulate swarm ────────────────────────────────────────────
    print(f"\n[5/6] Simulating swarm  (N={2*n_per_dir}, T={T_SIM:.0f} s, Δt={DT} s) …")
    t_arr, history, N = simulate_swarm(spline, n_per_dir=n_per_dir)

    print("\n  Computing collision metrics …")
    min_dists, violations = compute_metrics(history, N, spline)
    n_coll = int((min_dists < 2 * ROBOT_R).sum())
    t_near = float((min_dists < RSAFE).sum()) * DT
    print(f"  Hard collisions (d < 2R={2*ROBOT_R:.2f}m) : {n_coll} steps")
    print(f"  Time in danger zone (d < Rsafe)           : {t_near:.2f} s")
    print(f"  Min distance ever                         : {min_dists.min():.4f} m")
    plot_metrics(t_arr, min_dists, violations, n_per_dir)

    # ── Step 6: Animate ───────────────────────────────────────────────────
    print("\n[6/6] Rendering animation …")
    animate_swarm(rgb, spline, t_arr, history, N, n_per_dir)

    print("\n✓  Task 2 complete!")
    print("   Outputs: task2_pipeline.png   task2_metrics.png   task2_swarm.gif")


if __name__ == "__main__":
    main()