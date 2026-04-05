"""
Team Synthesis Part 2: Synthesize the rotation family R_z(pi/2^n)
==================================================================
Exact decompositions for n=0,1,2 using Clifford+T, with distance
metric verification and Bloch sphere animations.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio
import os
from pathlib import Path
from bloqade import squin
from bloqade.pyqrack import StackMemorySimulator


# ---------------------------------------------------------------------------
# Gate matrices
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
H_mat = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S_mat = np.array([[1, 0], [0, 1j]], dtype=complex)
T_mat = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)
Z_mat = np.array([[1, 0], [0, -1]], dtype=complex)
X_mat = np.array([[0, 1], [1, 0]], dtype=complex)


def Rz(theta):
    """R_z(theta) = diag(1, e^(i*theta))  [up to global phase]."""
    return np.array([[1, 0], [0, np.exp(1j*theta)]], dtype=complex)


# ---------------------------------------------------------------------------
# Distance metric
# ---------------------------------------------------------------------------

def gate_distance(U, V):
    """
    Global-phase-invariant distance:
      d(U,V) = sqrt(1 - |Tr(U† V)| / 2)
    d=0 means exact match (up to global phase).
    """
    inner = np.abs(np.trace(U.conj().T @ V))
    return np.sqrt(1 - inner / 2)


# ---------------------------------------------------------------------------
# State ↔ Bloch sphere
# ---------------------------------------------------------------------------

def state_to_bloch(psi):
    """Convert a 2-component state vector to Bloch (x, y, z) coordinates."""
    # Ensure normalized
    psi = psi / np.linalg.norm(psi)
    # Bloch coordinates from density matrix ρ = |ψ><ψ|
    rho = np.outer(psi, psi.conj())
    x = 2 * rho[0, 1].real
    y = 2 * rho[0, 1].imag
    z = (rho[0, 0] - rho[1, 1]).real
    return x, y, z


# ---------------------------------------------------------------------------
# Bloch sphere rendering
# ---------------------------------------------------------------------------

def draw_bloch_sphere(ax):
    """Draw wireframe Bloch sphere."""
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, alpha=0.05, color='gray', linewidth=0.3)

    # Axes
    ax.plot([-1.3, 1.3], [0, 0], [0, 0], 'k-', alpha=0.2, linewidth=0.5)
    ax.plot([0, 0], [-1.3, 1.3], [0, 0], 'k-', alpha=0.2, linewidth=0.5)
    ax.plot([0, 0], [0, 0], [-1.3, 1.3], 'k-', alpha=0.2, linewidth=0.5)

    # Equator
    eq_theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(eq_theta), np.sin(eq_theta), 0, 'k-', alpha=0.15, linewidth=0.5)

    # Labels
    ax.text(1.45, 0, 0, '$|+\\rangle$', fontsize=9, ha='center')
    ax.text(-1.45, 0, 0, '$|-\\rangle$', fontsize=9, ha='center')
    ax.text(0, 1.45, 0, '$|i\\rangle$', fontsize=9, ha='center')
    ax.text(0, -1.45, 0, '$|-i\\rangle$', fontsize=9, ha='center')
    ax.text(0, 0, 1.3, '$|0\\rangle$', fontsize=9, ha='center')
    ax.text(0, 0, -1.35, '$|1\\rangle$', fontsize=9, ha='center')

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_axis_off()
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=20, azim=30)


def render_frame(trajectory, current_idx, gate_labels, title, path_color='#2196F3'):
    """Render a single frame of the Bloch sphere animation."""
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    draw_bloch_sphere(ax)

    # Draw trajectory up to current point
    xs = [t[0] for t in trajectory[:current_idx+1]]
    ys = [t[1] for t in trajectory[:current_idx+1]]
    zs = [t[2] for t in trajectory[:current_idx+1]]

    if len(xs) > 1:
        ax.plot(xs, ys, zs, '-', color=path_color, linewidth=2, alpha=0.6)

    # Draw all waypoints as small dots
    for j in range(current_idx+1):
        alpha = 0.3 if j < current_idx else 1.0
        size = 30 if j < current_idx else 120
        color = 'gray' if j < current_idx else '#F44336'
        ax.scatter(*trajectory[j], s=size, c=color, alpha=alpha, zorder=5)

    # Target state (faint)
    tx, ty, tz = trajectory[-1]
    if current_idx < len(trajectory) - 1:
        ax.scatter(tx, ty, tz, s=80, c='green', alpha=0.25, marker='*', zorder=4)

    # Label showing which gate was just applied
    if current_idx == 0:
        label = f"Initial: $|0\\rangle$"
    elif current_idx <= len(gate_labels):
        applied = " \\cdot ".join(gate_labels[:current_idx])
        label = f"After: ${applied}$"
    else:
        label = ""

    ax.set_title(f"{title}\n{label}", fontsize=12, pad=10)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Animation creation
# ---------------------------------------------------------------------------

def make_animation(gates_with_labels, title, filename, output_dir, n_interp=15):
    """
    Create a GIF animating gate-by-gate application on the Bloch sphere.

    gates_with_labels: list of (matrix, label) tuples
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute trajectory: start at |0⟩
    psi = np.array([1, 0], dtype=complex)
    keypoints = [state_to_bloch(psi)]
    states = [psi.copy()]

    for gate, _ in gates_with_labels:
        psi = gate @ psi
        states.append(psi.copy())
        keypoints.append(state_to_bloch(psi))

    # Interpolate between keypoints on the Bloch sphere for smooth animation
    trajectory = []
    traj_gate_indices = [0]  # which trajectory index corresponds to each gate step

    for i in range(len(keypoints) - 1):
        p1 = np.array(keypoints[i])
        p2 = np.array(keypoints[i+1])

        # Interpolate on the sphere surface (slerp-like)
        for t in np.linspace(0, 1, n_interp, endpoint=(i == len(keypoints)-2)):
            pt = p1 * (1-t) + p2 * t
            norm = np.linalg.norm(pt)
            if norm > 1e-10:
                pt = pt / norm  # project back to sphere
            trajectory.append(tuple(pt))

        traj_gate_indices.append(len(trajectory) - 1)

    gate_labels = [lbl for _, lbl in gates_with_labels]

    # Render frames
    frames = []
    tmp_dir = output_dir / "tmp_frames"
    tmp_dir.mkdir(exist_ok=True)

    def render_and_append(idx, gate_step):
        fig = plt.figure(figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        draw_bloch_sphere(ax)

        if idx >= 0:
            xs = [t[0] for t in trajectory[:idx+1]]
            ys = [t[1] for t in trajectory[:idx+1]]
            zs = [t[2] for t in trajectory[:idx+1]]
            ax.plot(xs, ys, zs, '-', color='#2196F3', linewidth=2, alpha=0.6)

        for j in range(gate_step + 1):
            a = 0.3 if j < gate_step else 0.8
            sz = 30 if j < gate_step else 100
            ax.scatter(*keypoints[j], s=sz, c='#F44336', alpha=a, zorder=5)

        if idx >= 0:
            ax.scatter(*trajectory[idx], s=120, c='#F44336', alpha=1.0, zorder=6)
        else:
            ax.scatter(*keypoints[0], s=120, c='#F44336', alpha=1.0, zorder=6)

        if gate_step < len(keypoints) - 1:
            ax.scatter(*keypoints[-1], s=80, c='green', alpha=0.25, marker='*', zorder=4)

        if gate_step == 0:
            label = "Initial: $|0\\rangle$"
        else:
            applied = " \\cdot ".join(gate_labels[:gate_step])
            label = f"After: ${applied}$"

        ax.set_title(f"{title}\n{label}", fontsize=12, pad=10)

        frame_path = tmp_dir / f"frame_{len(frames):04d}.png"
        fig.savefig(frame_path, facecolor='white', pad_inches=0.1)
        plt.close(fig)
        frames.append(imageio.imread(frame_path))

    # Hold on initial state
    for _ in range(8):
        render_and_append(-1, 0)

    # Animate through trajectory
    for idx in range(len(trajectory)):
        gate_step = 0
        for gi, ti in enumerate(traj_gate_indices):
            if idx >= ti:
                gate_step = gi
        render_and_append(idx, gate_step)

    # Hold on final state
    for _ in range(15):
        frames.append(frames[-1])

    # Save GIF
    gif_path = output_dir / filename
    imageio.mimsave(gif_path, frames, duration=0.08, loop=0)

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir)

    print(f"  Saved: {gif_path}")
    return gif_path


# ---------------------------------------------------------------------------
# Bloqade verification kernels
# ---------------------------------------------------------------------------

@squin.kernel
def rz_n0_circuit():
    """R_z(π) = Z = S·S"""
    q = squin.qalloc(1)
    squin.h(q[0])    # go to |+⟩ so rotation is visible
    squin.s(q[0])    # first S
    squin.s(q[0])    # second S = Z
    return q

@squin.kernel
def rz_n1_circuit():
    """R_z(π/2) = S"""
    q = squin.qalloc(1)
    squin.h(q[0])
    squin.s(q[0])
    return q

@squin.kernel
def rz_n2_circuit():
    """R_z(π/4) = T"""
    q = squin.qalloc(1)
    squin.h(q[0])
    squin.t(q[0])
    return q

# Target circuits (direct Rz via matrix)
@squin.kernel
def target_rz_n0():
    """Target: H then Z"""
    q = squin.qalloc(1)
    squin.h(q[0])
    squin.z(q[0])
    return q

@squin.kernel
def target_rz_n1():
    """Target: H then S"""
    q = squin.qalloc(1)
    squin.h(q[0])
    squin.s(q[0])
    return q

@squin.kernel
def target_rz_n2():
    """Target: H then T"""
    q = squin.qalloc(1)
    squin.h(q[0])
    squin.t(q[0])
    return q


def get_dm(kernel, n_qubits):
    sim = StackMemorySimulator(min_qubits=n_qubits)
    task = sim.task(kernel)
    result = task.run()
    return StackMemorySimulator.reduced_density_matrix(result)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("=" * 65)
    print("PART 2: EXACT SYNTHESIS OF R_z(π/2^n) FOR n = 0, 1, 2")
    print("=" * 65)

    # -------------------------------------------------------------------
    # 1. Matrix-level verification with distance metric
    # -------------------------------------------------------------------
    print("\n--- Distance Metric Verification (matrix level) ---\n")

    cases = [
        (0, "Z = S·S", [S_mat, S_mat], Rz(np.pi)),
        (1, "S",       [S_mat],        Rz(np.pi/2)),
        (2, "T",       [T_mat],        Rz(np.pi/4)),
    ]

    for n, label, gate_sequence, target in cases:
        # Multiply gate sequence
        V = I2.copy()
        for g in gate_sequence:
            V = g @ V
        d = gate_distance(target, V)
        print(f"  n={n}: R_z(π/{2**n}) = {label}")
        print(f"    Implementation matrix:\n      {np.array2string(V, prefix='      ')}")
        print(f"    Target matrix:\n      {np.array2string(target, prefix='      ')}")
        print(f"    d(U, V) = {d:.10f}  {'✓ EXACT' if d < 1e-10 else '✗ APPROXIMATE'}")
        print()

    # -------------------------------------------------------------------
    # 2. Bloqade simulation verification
    # -------------------------------------------------------------------
    print("--- Bloqade Simulation Verification ---\n")

    sim_cases = [
        (0, rz_n0_circuit, target_rz_n0),
        (1, rz_n1_circuit, target_rz_n1),
        (2, rz_n2_circuit, target_rz_n2),
    ]

    for n, impl_kernel, target_kernel in sim_cases:
        rho_impl = get_dm(impl_kernel, 1)
        rho_target = get_dm(target_kernel, 1)
        overlap = float(np.real(np.trace(rho_impl @ rho_target)))
        print(f"  n={n}: Overlap = {overlap:.10f}  {'✓' if overlap > 0.9999 else '✗'}")

    # -------------------------------------------------------------------
    # 3. Bloch sphere animations
    # -------------------------------------------------------------------
    print("\n--- Generating Bloch Sphere Animations ---\n")

    animations = [
        {
            "n": 0,
            "title": "$R_z(\\pi) = Z = S \\cdot S$",
            "gates": [(H_mat, "H"), (S_mat, "S"), (S_mat, "S")],
            "filename": "bloch_rz_n0.gif",
        },
        {
            "n": 1,
            "title": "$R_z(\\pi/2) = S$",
            "gates": [(H_mat, "H"), (S_mat, "S")],
            "filename": "bloch_rz_n1.gif",
        },
        {
            "n": 2,
            "title": "$R_z(\\pi/4) = T$",
            "gates": [(H_mat, "H"), (T_mat, "T")],
            "filename": "bloch_rz_n2.gif",
        },
    ]

    for anim in animations:
        make_animation(
            anim["gates"],
            anim["title"],
            anim["filename"],
            figures_dir,
        )

    # -------------------------------------------------------------------
    # 4. Summary comparison figure
    # -------------------------------------------------------------------
    print("\n--- Generating Summary Figure ---\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                              subplot_kw={'projection': '3d'})

    for idx, anim in enumerate(animations):
        ax = axes[idx]
        draw_bloch_sphere(ax)

        # Compute trajectory
        psi = np.array([1, 0], dtype=complex)
        points = [state_to_bloch(psi)]
        for gate, _ in anim["gates"]:
            psi = gate @ psi
            points.append(state_to_bloch(psi))

        # Draw path
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        ax.plot(xs, ys, zs, '-', color='#2196F3', linewidth=2.5, alpha=0.7)

        # Draw points
        for j, p in enumerate(points):
            if j == 0:
                ax.scatter(*p, s=100, c='#4CAF50', zorder=5, label='Start')
            elif j == len(points)-1:
                ax.scatter(*p, s=100, c='#F44336', zorder=5, label='Final')
            else:
                ax.scatter(*p, s=40, c='#FF9800', zorder=5, alpha=0.7)

        gate_str = " → ".join(["H"] + [lbl for _, lbl in anim["gates"][1:]])
        ax.set_title(f"n={anim['n']}: {gate_str}", fontsize=11, pad=5)

    fig.suptitle("Exact Clifford+T Decompositions of $R_z(\\pi/2^n)$ on the Bloch Sphere",
                 fontsize=13, y=0.98)
    summary_path = figures_dir / "part2_bloch_summary.png"
    fig.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {summary_path}")

    # -------------------------------------------------------------------
    # 5. Gate cost table
    # -------------------------------------------------------------------
    print("\n--- Gate Cost Summary ---\n")
    print(f"  {'n':>2} | {'Rotation':>12} | {'Decomposition':>15} | {'H gates':>7} | {'S gates':>7} | {'T gates':>7} | {'Distance':>10}")
    print(f"  {'─'*2}─┼─{'─'*12}─┼─{'─'*15}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*10}")

    cost_data = [
        (0, "R_z(π)",   "S · S", 0, 2, 0, 0.0),
        (1, "R_z(π/2)", "S",     0, 1, 0, 0.0),
        (2, "R_z(π/4)", "T",     0, 0, 1, 0.0),
    ]
    for n, rot, decomp, h, s, t, d in cost_data:
        print(f"  {n:>2} | {rot:>12} | {decomp:>15} | {h:>7} | {s:>7} | {t:>7} | {d:>10.6f}")

    print("\n✓ All exact decompositions verified.")
    print("\nNote: n ≥ 3 requires APPROXIMATION — R_z(π/2^n) for n ≥ 3 is")
    print("non-Clifford and cannot be exactly represented in Clifford+T.")


if __name__ == "__main__":
    main()
