"""Generate Bloch sphere figures showing H, S, T gate rotation axes and actions."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
from mpl_toolkits.mplot3d import Axes3D

def draw_bloch_sphere(ax, title=""):
    """Draw a wireframe Bloch sphere."""
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.05, color='lightblue', edgecolor='none')

    # Equator and meridians
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 0, color='gray', alpha=0.3, lw=0.5)
    ax.plot(np.cos(theta), np.zeros_like(theta), np.sin(theta), color='gray', alpha=0.3, lw=0.5)
    ax.plot(np.zeros_like(theta), np.cos(theta), np.sin(theta), color='gray', alpha=0.3, lw=0.5)

    # Axes
    ax.plot([-1.3, 1.3], [0, 0], [0, 0], 'k-', alpha=0.2, lw=0.5)
    ax.plot([0, 0], [-1.3, 1.3], [0, 0], 'k-', alpha=0.2, lw=0.5)
    ax.plot([0, 0], [0, 0], [-1.3, 1.3], 'k-', alpha=0.2, lw=0.5)

    # Labels
    ax.text(1.3, 0, 0, r'$x$', fontsize=16)
    ax.text(0, 1.3, 0, r'$y$', fontsize=16)
    ax.text(0, 0, 1.3, r'$z$', fontsize=16)

    # Key states
    ax.scatter([0, 0, 0, 0, 1, -1], [0, 0, 1, -1, 0, 0], [1, -1, 0, 0, 0, 0],
               color='black', s=15, zorder=5)
    ax.text(0.08, 0, 1.15, r'$|0\rangle$', fontsize=14)
    ax.text(0.08, 0, -1.25, r'$|1\rangle$', fontsize=14)
    ax.text(1.15, 0, 0.1, r'$|+\rangle$', fontsize=14)
    ax.text(-1.45, 0, 0.1, r'$|-\rangle$', fontsize=14)

    lim = 1.2
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, fontweight='bold', pad=-10)
    ax.view_init(elev=20, azim=-60)


def draw_rotation_arc(ax, axis_vec, angle, start_point, color, n_pts=50):
    """Draw an arc showing rotation of start_point around axis_vec by angle."""
    axis_vec = np.array(axis_vec, dtype=float)
    axis_vec /= np.linalg.norm(axis_vec)
    start_point = np.array(start_point, dtype=float)

    points = []
    for t in np.linspace(0, angle, n_pts):
        # Rodrigues' rotation formula
        p = (start_point * np.cos(t)
             + np.cross(axis_vec, start_point) * np.sin(t)
             + axis_vec * np.dot(axis_vec, start_point) * (1 - np.cos(t)))
        points.append(p)
    points = np.array(points)
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, lw=2.5, alpha=0.8)
    # Arrow at end
    ax.scatter(*points[-1], color=color, s=40, zorder=5, marker='o')
    return points[-1]


def draw_axis_arrow(ax, axis_vec, color, label, length=1.25):
    """Draw the rotation axis as a colored arrow."""
    axis_vec = np.array(axis_vec, dtype=float)
    axis_vec = axis_vec / np.linalg.norm(axis_vec) * length
    ax.quiver(0, 0, 0, axis_vec[0], axis_vec[1], axis_vec[2],
              color=color, arrow_length_ratio=0.1, lw=2.5, alpha=0.9)
    ax.text(axis_vec[0]*1.15, axis_vec[1]*1.15, axis_vec[2]*1.15,
            label, fontsize=15, color=color, fontweight='bold')


# ============================================================
# Figure 1: Individual gate Bloch spheres (H, S, T)
# ============================================================
fig = plt.figure(figsize=(15, 5))
fig.subplots_adjust(left=-0.12, right=1.12, top=1.1, bottom=-0.1, wspace=-0.25)

# --- T gate: π/4 rotation about z ---
ax1 = fig.add_subplot(131, projection='3d')
draw_bloch_sphere(ax1, '')
draw_axis_arrow(ax1, [0, 0, 1], '#d62728', r'$\hat{z}$')
start = np.array([1, 0, 0])  # |+>
end = draw_rotation_arc(ax1, [0, 0, 1], np.pi/4, start, '#d62728')
ax1.scatter(*start, color='#2ca02c', s=60, zorder=6, marker='*')
ax1.set_box_aspect([1, 1, 1])

# --- S gate: π/2 rotation about z ---
ax2 = fig.add_subplot(132, projection='3d')
draw_bloch_sphere(ax2, '')
draw_axis_arrow(ax2, [0, 0, 1], '#1f77b4', r'$\hat{z}$')
start = np.array([1, 0, 0])
end = draw_rotation_arc(ax2, [0, 0, 1], np.pi/2, start, '#1f77b4')
ax2.scatter(*start, color='#2ca02c', s=60, zorder=6, marker='*')
ax2.set_box_aspect([1, 1, 1])

# --- H gate: π rotation about (x+z)/√2 ---
ax3 = fig.add_subplot(133, projection='3d')
h_axis = np.array([1, 0, 1]) / np.sqrt(2)
draw_bloch_sphere(ax3, '')
draw_axis_arrow(ax3, h_axis, '#9467bd', r'$\frac{\hat{x}+\hat{z}}{\sqrt{2}}$')
start = np.array([0, 0, 1])  # |0>
end = draw_rotation_arc(ax3, h_axis, np.pi, start, '#9467bd')
ax3.scatter(*start, color='#2ca02c', s=60, zorder=6, marker='*')
ax3.set_box_aspect([1, 1, 1])

plt.savefig(FIGURES_DIR / 'bloch_gates_HST.png',
            dpi=400, facecolor='white')
plt.savefig(FIGURES_DIR / 'bloch_gates_HST.pdf',
            facecolor='white')
print("Saved bloch_gates_HST.png/pdf")
plt.close()

# ============================================================
# Figure 2: Why no finite exact universal set (continuous Bloch sphere)
# ============================================================
fig2 = plt.figure(figsize=(5, 5))
ax = fig2.add_subplot(111, projection='3d', computed_zorder=False)
fig2.subplots_adjust(left=-0.25, right=1.25, top=1.15, bottom=-0.15)
draw_bloch_sphere(ax, '')

# Scatter many points on the sphere to show density of possible states
np.random.seed(42)
n_pts = 200
phi = np.random.uniform(0, 2*np.pi, n_pts)
costheta = np.random.uniform(-1, 1, n_pts)
theta = np.arccos(costheta)
xs = np.sin(theta) * np.cos(phi)
ys = np.sin(theta) * np.sin(phi)
zs = np.cos(theta)
colors = plt.cm.hsv(phi / (2*np.pi))
ax.scatter(xs, ys, zs, c=colors, s=8, alpha=0.6, zorder=5)

# Compute ACTUAL reachable Bloch vectors from Clifford+T sequences up to depth ~4
# Gates as 2x2 matrices acting on state vectors
H_m = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)
T_m = np.diag([1, np.exp(1j*np.pi/4)])
Td_m = np.diag([1, np.exp(-1j*np.pi/4)])  # T-dagger
gates = [H_m, T_m, Td_m]

def state_to_bloch(psi):
    """Convert |psi> to Bloch (x,y,z)."""
    rho = np.outer(psi, psi.conj())
    x = 2*np.real(rho[0,1])
    y = 2*np.imag(rho[1,0])
    z = np.real(rho[0,0] - rho[1,1])
    return (x, y, z)

# BFS over gate sequences up to depth 8 starting from |0> and |1>
bloch_pts = set()
starts = [np.array([1,0], dtype=complex), np.array([0,1], dtype=complex)]
current_states = list(starts)
for depth in range(9):
    next_states = []
    for psi in current_states:
        bv = state_to_bloch(psi)
        rounded = (round(bv[0], 4), round(bv[1], 4), round(bv[2], 4))
        if rounded not in bloch_pts:
            bloch_pts.add(rounded)
            for g in gates:
                next_states.append(g @ psi)
    current_states = next_states
    if len(bloch_pts) > 500:  # cap for performance
        break

reachable = np.array(list(bloch_pts))
ax.scatter(reachable[:,0], reachable[:,1], reachable[:,2],
           color='red', s=25, zorder=10, marker='D', edgecolors='black', linewidth=0.3, alpha=0.8)

# Force equal aspect ratio so sphere looks round
ax.set_box_aspect([1, 1, 1])

plt.savefig(FIGURES_DIR / 'bloch_continuous.png',
            dpi=400, facecolor='white')
plt.savefig(FIGURES_DIR / 'bloch_continuous.pdf',
            facecolor='white')
print("Saved bloch_continuous.png/pdf")
plt.close()

# ============================================================
# Figure 3: CNOT action (truth table style, not Bloch)
# ============================================================
fig3, ax = plt.subplots(figsize=(6, 3))
ax.set_axis_off()

table_data = [
    [r'$|00\rangle$', r'$|00\rangle$'],
    [r'$|01\rangle$', r'$|01\rangle$'],
    [r'$|10\rangle$', r'$|11\rangle$'],
    [r'$|11\rangle$', r'$|10\rangle$'],
]
table = ax.table(cellText=table_data,
                 colLabels=['Input', 'Output'],
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1.2, 1.8)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#4C72B0')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('#f0f0f0' if row % 2 == 0 else 'white')

ax.set_title(r'CNOT: flips target iff control $= |1\rangle$', fontsize=13, pad=20)
plt.savefig(FIGURES_DIR / 'cnot_truth_table.png',
            dpi=400, bbox_inches='tight', facecolor='white')
plt.savefig(FIGURES_DIR / 'cnot_truth_table.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved cnot_truth_table.png/pdf")
plt.close()

print("All figures generated!")
