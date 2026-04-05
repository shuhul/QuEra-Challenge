"""
Part 2 (continued): Approximate synthesis of R_z(pi/2^n) for n >= 3
=====================================================================
Compares Solovay-Kitaev (implemented from scratch) with
Ross-Selinger gridsynth (via pygridsynth package).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio
import shutil
from pathlib import Path
from collections import deque
import time
import mpmath
from pygridsynth.gridsynth import gridsynth_gates

# ===========================================================================
# Gate matrices
# ===========================================================================

I2 = np.eye(2, dtype=complex)
H_mat = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
T_mat = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
Td_mat = T_mat.conj().T  # T-dagger
S_mat = T_mat @ T_mat     # S = T^2
Sd_mat = Td_mat @ Td_mat  # S-dagger

GENERATORS = [H_mat, T_mat, Td_mat]
GATE_NAMES = ['H', 'T', 'Td']
GATE_INV = {'H': 'H', 'T': 'Td', 'Td': 'T', 'S': 'Sd', 'Sd': 'S', 'X': 'X'}

X_mat = np.array([[0, 1], [1, 0]], dtype=complex)

ALL_GATE_MATRICES = {
    'H': H_mat, 'T': T_mat, 'Td': Td_mat, 'S': S_mat, 'Sd': Sd_mat, 'X': X_mat
}


def Rz(theta):
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)


def gate_distance(U, V):
    """Global-phase-invariant distance: d = sqrt(1 - |Tr(U^dag V)|/2)."""
    inner = np.abs(np.trace(U.conj().T @ V))
    return np.sqrt(max(0.0, 1.0 - inner / 2.0))


def to_su2(U):
    """Project onto SU(2) by removing global phase."""
    d = np.linalg.det(U)
    phase = np.sqrt(d)
    if abs(phase) < 1e-15:
        return U
    return U / phase


def mat_hash(U, precision=10):
    """Hash a unitary by its SU(2) projection, rounded to given precision."""
    U_n = to_su2(U)
    return tuple(np.round(
        np.concatenate([U_n.real.flatten(), U_n.imag.flatten()]),
        precision
    ))


def seq_to_unitary(seq):
    """Multiply out a gate sequence (applied left to right)."""
    U = I2.copy()
    for g in seq:
        U = U @ ALL_GATE_MATRICES[g]
    return U


def inverse_sequence(seq):
    """Return the gate sequence for U-dagger."""
    return [GATE_INV[g] for g in reversed(seq)]


def t_count(seq):
    """Count T and T-dagger gates in a sequence."""
    return sum(1 for g in seq if g in ('T', 'Td'))


# ===========================================================================
# Bloch sphere utilities
# ===========================================================================

def state_to_bloch(psi):
    psi = psi / np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    x = 2 * rho[0, 1].real
    y = -2 * rho[0, 1].imag
    z = (rho[0, 0] - rho[1, 1]).real
    return np.array([x, y, z])


# ===========================================================================
# SOLOVAY-KITAEV (implemented from scratch)
# ===========================================================================

class SolovayKitaev:
    """
    Solovay-Kitaev algorithm for approximating single-qubit unitaries.

    1. Build a database of all short {H, T, T-dagger} sequences via BFS
    2. Depth 0: return the closest database entry
    3. Depth n:
       a. Get depth-(n-1) approximation U_prev
       b. Compute error Delta = target * U_prev-dagger
       c. Decompose Delta = V W V-dag W-dag via balanced group commutator
       d. Recursively approximate V and W at depth (n-1)
       e. Return V_approx * W_approx * V_approx-dag * W_approx-dag * U_prev
    """

    def __init__(self, db_max_depth=15):
        print(f"  Building SK database (BFS up to depth {db_max_depth})...",
              end=" ", flush=True)
        t0 = time.time()
        self.database = self._build_database(db_max_depth)
        print(f"done ({len(self.database)} entries, {time.time()-t0:.1f}s)")

    def _build_database(self, max_depth):
        """BFS over {H, T, T-dagger} sequences up to max_depth."""
        db = []
        queue = deque([(I2.copy(), [])])
        seen = {mat_hash(I2)}

        while queue:
            U, seq = queue.popleft()
            if len(seq) > max_depth:
                continue
            db.append((U.copy(), list(seq)))
            for i in range(3):
                new_U = U @ GENERATORS[i]
                h = mat_hash(new_U)
                if h not in seen:
                    seen.add(h)
                    queue.append((new_U, seq + [GATE_NAMES[i]]))
        return db

    def basic_approx(self, target):
        """Find closest entry in the database."""
        best_d = float('inf')
        best_U, best_seq = I2, []
        for U, seq in self.database:
            d = gate_distance(target, U)
            if d < best_d:
                best_d = d
                best_U = U
                best_seq = seq
        return best_U, list(best_seq), best_d

    @staticmethod
    def _extract_rotation(U):
        """Extract rotation angle theta and axis n-hat from SU(2) matrix.

        SU(2) matrix: U = cos(theta/2) I + i sin(theta/2) (n-hat . sigma)
        Returns theta in [0, pi] and unit vector n-hat.
        """
        U_su2 = to_su2(U)
        # Choose the representative with theta in [0, pi]
        if U_su2[0, 0].real < 0:
            U_su2 = -U_su2

        cos_half = np.clip(U_su2[0, 0].real, 0.0, 1.0)
        theta = 2 * np.arccos(cos_half)

        if theta < 1e-12:
            return 0.0, np.array([0.0, 0.0, 1.0])

        s = np.sin(theta / 2)
        if abs(s) < 1e-12:
            return 0.0, np.array([0.0, 0.0, 1.0])

        # From U[1,0] = (i nx - ny) sin(theta/2)
        nx = U_su2[1, 0].imag / s
        ny = -U_su2[1, 0].real / s
        nz = U_su2[0, 0].imag / s
        n_hat = np.array([nx, ny, nz])
        norm = np.linalg.norm(n_hat)
        if norm < 1e-12:
            return theta, np.array([0.0, 0.0, 1.0])
        return theta, n_hat / norm

    @staticmethod
    def _make_rotation(theta, n_hat):
        """Build SU(2) rotation: cos(theta/2) I + i sin(theta/2) (n-hat . sigma)."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        nx, ny, nz = n_hat
        return np.array([
            [c + 1j * nz * s, (1j * nx + ny) * s],
            [(1j * nx - ny) * s, c - 1j * nz * s]
        ], dtype=complex)

    def group_commutator_decompose(self, U):
        """
        Balanced group commutator decomposition.
        Given U close to I, find V, W such that V W V-dag W-dag ~ U.

        Uses: for perpendicular axes s-hat, t-hat with s-hat x t-hat = n-hat,
        R_s(alpha) R_t(alpha) R_s(-alpha) R_t(-alpha) = R_n(theta)
        where cos(theta/2) = 1 - 2 sin^4(alpha/2).
        """
        theta, n_hat = self._extract_rotation(U)

        if theta < 1e-12:
            return I2.copy(), I2.copy()

        # Two perpendicular unit vectors orthogonal to n-hat
        if abs(n_hat[0]) < 0.9:
            s_hat = np.cross(n_hat, [1, 0, 0])
        else:
            s_hat = np.cross(n_hat, [0, 1, 0])
        s_hat /= np.linalg.norm(s_hat)
        t_hat = np.cross(n_hat, s_hat)
        t_hat /= np.linalg.norm(t_hat)

        # Solve for alpha: cos(theta/2) = 1 - 2 sin^4(alpha/2)
        cos_half = np.cos(theta / 2)
        val = max(0.0, (1.0 - cos_half) / 2.0)
        sin_alpha_half = np.clip(val ** 0.25, 0, 1)
        alpha = 2 * np.arcsin(sin_alpha_half)

        V = self._make_rotation(alpha, s_hat)
        W = self._make_rotation(alpha, t_hat)
        return V, W

    def solve(self, target, depth):
        """Run SK algorithm at given recursion depth.
        Returns: (unitary, gate_sequence, distance)
        """
        if depth == 0:
            return self.basic_approx(target)

        # Get depth-(n-1) approximation
        U_prev, seq_prev, d_prev = self.solve(target, depth - 1)

        # Already excellent — skip GCD
        if d_prev < 1e-10:
            return U_prev, seq_prev, d_prev

        # Error unitary (should be close to I)
        Delta = target @ U_prev.conj().T

        # Decompose error via group commutator
        V_exact, W_exact = self.group_commutator_decompose(Delta)

        # Recursively approximate V and W
        V_approx, seq_V, _ = self.solve(V_exact, depth - 1)
        W_approx, seq_W, _ = self.solve(W_exact, depth - 1)

        # Combine: W V W-dag V-dag U_prev  (note: [W,V] not [V,W])
        seq_Vd = inverse_sequence(seq_V)
        seq_Wd = inverse_sequence(seq_W)
        full_seq = seq_W + seq_V + seq_Wd + seq_Vd + seq_prev
        full_U = seq_to_unitary(full_seq)
        dist = gate_distance(target, full_U)

        # Fallback: if this depth is worse, keep the previous result
        if dist >= d_prev:
            return U_prev, seq_prev, d_prev

        return full_U, full_seq, dist


# ===========================================================================
# ROSS-SELINGER (gridsynth) via pygridsynth
# ===========================================================================

# pygridsynth gate char -> our gate name (W = global phase, skip for physical count)
PYGRID_GATE_MAP = {'H': 'H', 'T': 'T', 'S': 'S', 'X': 'X'}


def _gridsynth_to_seq(gs_str):
    """Convert pygridsynth string to our gate sequence (physical gates only)."""
    return [PYGRID_GATE_MAP[c] for c in gs_str if c in PYGRID_GATE_MAP]


def ross_selinger_search(theta_float, target, epsilons):
    """
    Run Ross-Selinger gridsynth (pygridsynth) at multiple epsilon values.
    Returns a list of dicts.
    """
    theta_mp = mpmath.mpf(theta_float)
    results = []
    for eps in epsilons:
        t0 = time.time()
        gs_str = gridsynth_gates(theta=theta_mp, epsilon=mpmath.mpf(eps))
        elapsed = time.time() - t0

        seq = _gridsynth_to_seq(gs_str)
        if not seq:
            # Empty result means identity is within epsilon (small angle)
            print(f"    eps={eps:.0e}: identity (angle within epsilon)")
            continue

        U = seq_to_unitary(seq)
        dist = gate_distance(target, U)
        tc = sum(1 for g in seq if g == 'T')
        total = len(seq)

        results.append({
            'epsilon': eps,
            'total_gates': total,
            't_count': tc,
            'distance': dist,
            'sequence': seq,
            'time': elapsed,
        })
        print(f"    eps={eps:.0e}: {total:>4} gates, T-count={tc:>3}, "
              f"d={dist:.2e}, time={elapsed:.2f}s")

    return results


# ===========================================================================
# Bloch sphere animation
# ===========================================================================

def draw_bloch_sphere(ax):
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, alpha=0.05, color='gray', linewidth=0.3)

    ax.plot([-1.3, 1.3], [0, 0], [0, 0], 'k-', alpha=0.2, lw=0.5)
    ax.plot([0, 0], [-1.3, 1.3], [0, 0], 'k-', alpha=0.2, lw=0.5)
    ax.plot([0, 0], [0, 0], [-1.3, 1.3], 'k-', alpha=0.2, lw=0.5)

    eq = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(eq), np.sin(eq), 0, 'k-', alpha=0.15, lw=0.5)

    ax.text(1.5, 0, 0, '$|+\\rangle$', fontsize=8, ha='center')
    ax.text(-1.5, 0, 0, '$|-\\rangle$', fontsize=8, ha='center')
    ax.text(0, 1.5, 0, '$|i\\rangle$', fontsize=8, ha='center')
    ax.text(0, -1.5, 0, '$|-i\\rangle$', fontsize=8, ha='center')
    ax.text(0, 0, 1.3, '$|0\\rangle$', fontsize=8, ha='center')
    ax.text(0, 0, -1.35, '$|1\\rangle$', fontsize=8, ha='center')

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=30)


def make_convergence_gif(approximations, target_unitary, title, filepath):
    """Create a GIF showing convergence: one frame per approximation level.

    approximations: list of dicts with 'sequence', 'label', 'distance'
    Each frame shows H|0> = |+>, then the full approximation applied,
    so each successive frame gets closer to the target (green star).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Target Bloch point: H then target rotation on |0>
    plus = H_mat @ np.array([1, 0], dtype=complex)
    target_psi = target_unitary @ plus
    target_bloch = state_to_bloch(target_psi)
    start_bloch = state_to_bloch(plus)

    # Compute final Bloch point for each approximation level
    approx_points = []
    for approx in approximations:
        psi = plus.copy()
        for g in approx['sequence']:
            psi = ALL_GATE_MATRICES[g] @ psi
        approx_points.append(state_to_bloch(psi))

    tmp_dir = filepath.parent / f"tmp_{filepath.stem}"
    tmp_dir.mkdir(exist_ok=True)
    frames = []

    for i, (approx, pt) in enumerate(zip(approximations, approx_points)):
        fig = plt.figure(figsize=(6, 6), dpi=80)
        ax = fig.add_subplot(111, projection='3d')
        draw_bloch_sphere(ax)

        # Show target (green star)
        ax.scatter(*target_bloch, s=200, c='#4CAF50', marker='*', zorder=5,
                   label='Target')

        # Show start state |+>
        ax.scatter(*start_bloch, s=60, c='gray', alpha=0.4, zorder=4,
                   label='$|+\\rangle$')

        # Show all previous approximations as faded dots
        for j in range(i):
            ax.scatter(*approx_points[j], s=40, c='#FF9800', alpha=0.3, zorder=4)

        # Show current approximation (red dot)
        ax.scatter(*pt, s=120, c='#F44336', zorder=6, label='Approx')

        # Line from current to target
        ax.plot([pt[0], target_bloch[0]], [pt[1], target_bloch[1]],
                [pt[2], target_bloch[2]], '--', color='red', alpha=0.4, lw=1)

        bloch_d = np.linalg.norm(pt - target_bloch)
        ax.set_title(f"{title}\n{approx['label']}\n"
                     f"d={approx['distance']:.4f}, Bloch dist={bloch_d:.4f}",
                     fontsize=10, pad=5)
        ax.legend(fontsize=7, loc='upper left')

        fpath = tmp_dir / f"frame_{i:04d}.png"
        fig.savefig(fpath, facecolor='white', pad_inches=0.1)
        plt.close(fig)
        frames.append(imageio.imread(fpath))

    # Hold each frame longer, especially first and last
    padded = [frames[0]] * 4
    for f in frames:
        padded.extend([f] * 3)
    padded.extend([frames[-1]] * 6)

    imageio.mimsave(filepath, padded, duration=0.4, loop=0)
    shutil.rmtree(tmp_dir)
    print(f"    Saved GIF: {filepath} ({len(padded)} frames)")


# ===========================================================================
# Comparison plots
# ===========================================================================

def make_comparison_plot(sk_results, rs_results, filepath):
    """Plot total gate count vs distance for SK and Ross-Selinger."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, n in enumerate([3, 4, 5]):
        ax = axes[idx]

        # SK results
        sk_data = sk_results[n]
        sk_gates = [r['total_gates'] for r in sk_data]
        sk_dists = [max(r['distance'], 1e-16) for r in sk_data]

        ax.plot(sk_gates, sk_dists, 'rs-', markersize=8, linewidth=2,
                label='Solovay-Kitaev', zorder=5)
        for i, r in enumerate(sk_data):
            ax.annotate(f"d={r['depth']}", (sk_gates[i], sk_dists[i]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color='red')

        # Ross-Selinger results
        rs_data = rs_results[n]
        rs_gates = [r['total_gates'] for r in rs_data]
        rs_dists = [max(r['distance'], 1e-16) for r in rs_data]

        ax.plot(rs_gates, rs_dists, 'bo-', markersize=6, linewidth=2,
                label='Ross-Selinger', zorder=4)

        ax.set_yscale('log')
        ax.set_xlabel('Total gate count', fontsize=11)
        ax.set_title(f'$n={n}$: $R_z(\\pi/{2**n})$', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel('Distance $d(U,V)$  [log scale]', fontsize=11)
    fig.suptitle('Gate Count vs Approximation Quality:\n'
                 'Solovay-Kitaev vs Ross-Selinger',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved plot: {filepath}")


def make_tcount_comparison(sk_results, rs_results, filepath):
    """Plot T-count vs distance -- the key metric for fault-tolerant cost."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {3: '#E53935', 4: '#1E88E5', 5: '#43A047'}
    markers_sk = {3: 's', 4: 'D', 5: '^'}

    for n in [3, 4, 5]:
        # SK
        sk_tc = [r['t_count'] for r in sk_results[n]]
        sk_d = [max(r['distance'], 1e-16) for r in sk_results[n]]
        ax.plot(sk_tc, sk_d, marker=markers_sk[n], color=colors[n],
                linestyle='--', markersize=8, alpha=0.7,
                label=f'SK $n={n}$')

        # Ross-Selinger
        rs_tc = [r['t_count'] for r in rs_results[n]]
        rs_d = [max(r['distance'], 1e-16) for r in rs_results[n]]
        ax.plot(rs_tc, rs_d, marker='o', color=colors[n],
                linestyle='-', markersize=6,
                label=f'Ross-Selinger $n={n}$')

    ax.set_yscale('log')
    ax.set_xlabel('$T$-count (number of $T$ and $T^\\dagger$ gates)', fontsize=12)
    ax.set_ylabel('Distance $d(U,V)$  [log scale]', fontsize=12)
    ax.set_title('$T$-count vs Approximation Quality:\n'
                 'Solovay-Kitaev vs Ross-Selinger (gridsynth)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved plot: {filepath}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    figures_dir = Path(__file__).resolve().parent.parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    targets = {
        3: Rz(np.pi / 8),
        4: Rz(np.pi / 16),
        5: Rz(np.pi / 32),
    }

    print("=" * 65)
    print("PART 2: APPROXIMATE SYNTHESIS OF R_z(pi/2^n) FOR n = 3, 4, 5")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Solovay-Kitaev
    # ------------------------------------------------------------------
    print("\n--- Solovay-Kitaev Algorithm ---")
    sk = SolovayKitaev(db_max_depth=15)

    sk_results = {}
    for n, target in targets.items():
        print(f"\n  Target: R_z(pi/{2**n})  [n={n}]")
        sk_results[n] = []
        for depth in range(5):
            t0 = time.time()
            U, seq, dist = sk.solve(target, depth)
            elapsed = time.time() - t0
            tc = t_count(seq)
            print(f"    Depth {depth}: {len(seq):>5} gates, "
                  f"T-count={tc:>5}, d={dist:.10f}, time={elapsed:.2f}s")
            sk_results[n].append({
                'depth': depth,
                'total_gates': len(seq),
                't_count': tc,
                'distance': dist,
                'sequence': seq,
            })

    # ------------------------------------------------------------------
    # 2. Ross-Selinger (gridsynth via pygridsynth)
    # ------------------------------------------------------------------
    print("\n--- Ross-Selinger Algorithm (pygridsynth) ---")

    epsilons = [5e-2, 1e-2, 1e-3, 1e-6]

    rs_results = {}
    for n in [3, 4, 5]:
        theta = np.pi / (2 ** n)
        print(f"\n  Target: R_z(pi/{2**n})  [n={n}]")
        rs_results[n] = ross_selinger_search(theta, targets[n], epsilons)

    # ------------------------------------------------------------------
    # 3. Comparison table
    # ------------------------------------------------------------------
    print("\n--- Comparison Summary ---\n")
    print(f"  {'n':>2} | {'Method':>12} | {'Gates':>6} | {'T-cnt':>5} | "
          f"{'Distance':>14}")
    print(f"  {'─'*2}─┼─{'─'*12}─┼─{'─'*6}─┼─{'─'*5}─┼─{'─'*14}")

    for n in [3, 4, 5]:
        # Best SK
        best_sk = min(sk_results[n], key=lambda r: r['distance'])
        print(f"  {n:>2} | {'SK d=' + str(best_sk['depth']):>12} | "
              f"{best_sk['total_gates']:>6} | {best_sk['t_count']:>5} | "
              f"{best_sk['distance']:>14.10f}")

        if rs_results[n]:
            # RS at best precision
            rs_best = min(rs_results[n], key=lambda r: r['distance'])
            eps_str = f"{rs_best['epsilon']:.0e}"
            print(f"  {n:>2} | {'RS e=' + eps_str:>12} | "
                  f"{rs_best['total_gates']:>6} | {rs_best['t_count']:>5} | "
                  f"{rs_best['distance']:>14.10f}")
        else:
            print(f"  {n:>2} | {'RS':>12} | {'N/A':>6} | {'N/A':>5} | {'N/A':>14}")
        print(f"  {'─'*2}─┼─{'─'*12}─┼─{'─'*6}─┼─{'─'*5}─┼─{'─'*14}")

    # ------------------------------------------------------------------
    # 4. Convergence GIFs for n=3 and n=4
    # ------------------------------------------------------------------
    print("\n--- Generating Bloch Sphere Convergence GIFs ---")

    for n in [3, 4]:
        target_n = targets[n]

        # SK convergence: one frame per recursion depth
        sk_approxes = []
        for r in sk_results[n]:
            sk_approxes.append({
                'sequence': r['sequence'],
                'label': f"SK depth {r['depth']}: {r['total_gates']} gates",
                'distance': r['distance'],
            })
        make_convergence_gif(
            sk_approxes, target_n,
            f"SK Convergence: $R_z(\\pi/{2**n})$",
            figures_dir / f"bloch_sk_n{n}.gif",
        )

        # Ross-Selinger convergence: one frame per epsilon
        if rs_results[n]:
            rs_approxes = []
            for r in rs_results[n]:
                eps_str = f"{r['epsilon']:.0e}"
                rs_approxes.append({
                    'sequence': r['sequence'],
                    'label': f"RS eps={eps_str}: {r['total_gates']} gates, T={r['t_count']}",
                    'distance': r['distance'],
                })
            make_convergence_gif(
                rs_approxes, target_n,
                f"Ross-Selinger Convergence: $R_z(\\pi/{2**n})$",
                figures_dir / f"bloch_rs_n{n}.gif",
            )

    # ------------------------------------------------------------------
    # 5. Comparison plots
    # ------------------------------------------------------------------
    print("\n--- Generating Comparison Plots ---")
    make_comparison_plot(sk_results, rs_results,
                         figures_dir / "part2_sk_vs_rs_gates.png")
    make_tcount_comparison(sk_results, rs_results,
                            figures_dir / "part2_sk_vs_rs_tcount.png")

    print("\n✓ All approximate synthesis complete.")


if __name__ == "__main__":
    main()
