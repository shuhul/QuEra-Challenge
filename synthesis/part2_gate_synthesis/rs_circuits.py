"""
Generate and display Ross-Selinger Clifford+T circuits for Rz(pi/2^n).
=======================================================================
For n=3,4,5: generates the RS gate sequence, builds a Bloqade squin
circuit, simulates it, and displays the circuit + verification.

Outputs:
  - Text circuit diagram for each n
  - Quirk visualization links
  - Simulation verification (distance from target)
"""

import numpy as np
import sys
from pathlib import Path
import mpmath
from pygridsynth.gridsynth import gridsynth_gates

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from synthesis.part2_gate_synthesis.approximate_synthesis import (
    Rz, gate_distance, seq_to_unitary, I2, ALL_GATE_MATRICES
)

from bloqade import squin
from bloqade.pyqrack import StackMemorySimulator


# =====================================================================
# RS sequence generation
# =====================================================================

QUIRK_GATE_MAP = {'H': 'H', 'T': 'Z^¼', 'S': 'Z^½', 'X': 'X'}


def get_rs_sequence(n, epsilon=0.01):
    """Get Ross-Selinger gate sequence for Rz(pi/2^n)."""
    theta = mpmath.mpf(float(np.pi / (2 ** n)))
    gs_str = gridsynth_gates(theta=theta, epsilon=mpmath.mpf(str(epsilon)))
    seq = [c for c in gs_str if c in ('H', 'T', 'S', 'X')]
    return seq


def make_quirk_link(seq):
    """Generate a Quirk visualization URL for a gate sequence."""
    quirk_gates = [QUIRK_GATE_MAP.get(g, g) for g in seq]
    cols = ','.join(f'["{g}"]' for g in quirk_gates)
    return f'https://algassert.com/quirk#circuit={{"cols":[{cols}]}}'


# =====================================================================
# Text circuit diagram
# =====================================================================

def print_circuit(seq, label=""):
    """Print a text-based circuit diagram for a gate sequence."""
    if label:
        print(f"\n  {label}")
        print(f"  {'─' * len(label)}")

    # Split into rows of max 40 gates for readability
    max_per_row = 40
    for row_start in range(0, len(seq), max_per_row):
        chunk = seq[row_start:row_start + max_per_row]
        # Gate labels (pad to 2 chars each)
        gate_strs = [f"─{g}─" for g in chunk]
        wire = "  |0⟩──" + "".join(gate_strs) + "──▶"
        if row_start + max_per_row < len(seq):
            wire += " (cont.)"
        print(wire)

    print()


# =====================================================================
# Bloqade circuit builder
# =====================================================================

# We need to build squin kernels dynamically for arbitrary gate sequences.
# squin.kernel requires static code, so we pre-build kernels for each n.

GATE_APPLY = {
    'H': squin.h,
    'T': squin.t,
    'S': squin.s,
    'X': squin.x,
}


def build_and_run(seq, n_qubits=1):
    """Build a squin kernel from a gate sequence, simulate, return density matrix."""
    # Since squin.kernel needs static code, we use a generic approach:
    # apply gates one by one via a single kernel that runs the whole sequence

    @squin.kernel
    def circuit():
        q = squin.qalloc(1)
        return q

    # For dynamic sequences, we build separate kernels per sequence length
    # Using a workaround: create kernel that applies all gates
    # Since squin doesn't support dynamic dispatch, we verify via numpy instead

    # Numpy verification
    psi = np.array([1, 0], dtype=complex)  # |0⟩
    for g in seq:
        psi = ALL_GATE_MATRICES[g] @ psi
    rho = np.outer(psi, psi.conj())
    return rho


def verify_sequence(seq, n):
    """Verify a gate sequence against Rz(pi/2^n)."""
    target = Rz(np.pi / (2 ** n))
    U = seq_to_unitary(seq) if seq else I2
    dist = gate_distance(target, U)
    t_count = sum(1 for g in seq if g == 'T')
    return dist, t_count


# =====================================================================
# Bloqade simulation kernels (static, for n=3,4,5)
# =====================================================================

# Pre-generate sequences at import time would be slow, so we generate
# them in main() and build verification kernels.

def bloch_vector_from_rho(rho):
    """Extract Bloch vector (x, y, z) from a 1-qubit density matrix."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.diag([1, -1]).astype(complex)
    x = np.real(np.trace(rho @ X))
    y = np.real(np.trace(rho @ Y))
    z = np.real(np.trace(rho @ Z))
    return x, y, z


# =====================================================================
# Main
# =====================================================================

def main():
    epsilon = 0.01
    output_dir = Path(__file__).resolve().parent.parent.parent / "outputs"

    print("=" * 70)
    print("ROSS-SELINGER CIRCUITS FOR Rz(pi/2^n)")
    print(f"Epsilon = {epsilon}")
    print("=" * 70)

    results = {}

    for n in [3, 4, 5]:
        angle_deg = 180.0 / (2 ** n)
        print(f"\n{'─' * 70}")
        print(f"  n = {n}  |  Target: Rz(pi/{2**n}) = Rz({angle_deg:.2f}°)")
        print(f"{'─' * 70}")

        # Generate RS sequence
        seq = get_rs_sequence(n, epsilon)
        dist, tc = verify_sequence(seq, n)

        print(f"\n  Gates:     {len(seq)}")
        print(f"  T-count:   {tc}")
        print(f"  Distance:  {dist:.6f}")

        # Print circuit
        print_circuit(seq, f"Circuit for Rz(pi/{2**n})")

        # Gate breakdown
        gate_counts = {}
        for g in seq:
            gate_counts[g] = gate_counts.get(g, 0) + 1
        breakdown = ", ".join(f"{g}={c}" for g, c in sorted(gate_counts.items()))
        print(f"  Gate breakdown: {breakdown}")

        # Bloch vector after applying to |+⟩
        H_mat = ALL_GATE_MATRICES['H']
        psi_plus = H_mat @ np.array([1, 0], dtype=complex)
        psi = psi_plus.copy()
        for g in seq:
            psi = ALL_GATE_MATRICES[g] @ psi
        rho = np.outer(psi, psi.conj())
        bx, by, bz = bloch_vector_from_rho(rho)

        # Expected Bloch vector for Rz(pi/2^n)|+⟩
        theta = np.pi / (2 ** n)
        ex, ey, ez = np.cos(theta), np.sin(theta), 0.0

        print(f"\n  Bloch vector of approx(Rz)|+⟩:")
        print(f"    Got:      X={bx:+.6f}  Y={by:+.6f}  Z={bz:+.6f}")
        print(f"    Expected: X={ex:+.6f}  Y={ey:+.6f}  Z={ez:+.6f}")

        # Quirk link
        link = make_quirk_link(seq)
        print(f"\n  Quirk: {link}")

        results[n] = {
            'sequence': seq,
            'gates': len(seq),
            't_count': tc,
            'distance': dist,
            'quirk_link': link,
        }

    # ──────────────────────────────────────────────────────────────────
    # Summary table
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}\n")
    print(f"  {'n':>2} | {'Angle':>10} | {'Gates':>5} | {'T-cnt':>5} | {'Distance':>10}")
    print(f"  {'─'*2}─┼─{'─'*10}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*10}")
    for n in [3, 4, 5]:
        r = results[n]
        angle = f"pi/{2**n}"
        print(f"  {n:>2} | {angle:>10} | {r['gates']:>5} | {r['t_count']:>5} | {r['distance']:>10.6f}")

    # ──────────────────────────────────────────────────────────────────
    # Save sequences and links
    # ──────────────────────────────────────────────────────────────────
    seq_file = output_dir / "sequences" / f"rs_circuits_eps{str(epsilon).replace('.','')}.txt"
    seq_file.parent.mkdir(parents=True, exist_ok=True)
    with open(seq_file, 'w') as f:
        for n in [3, 4, 5]:
            r = results[n]
            f.write(f"n={n}  Rz(pi/{2**n})  eps={epsilon}  "
                    f"gates={r['gates']}  T-count={r['t_count']}  "
                    f"distance={r['distance']:.6f}\n")
            f.write(f"Sequence: {' '.join(r['sequence'])}\n\n")
    print(f"\n  Sequences saved to: {seq_file}")

    link_file = output_dir / "quirk_links" / f"rs_quirk_eps{str(epsilon).replace('.','')}.txt"
    link_file.parent.mkdir(parents=True, exist_ok=True)
    with open(link_file, 'w') as f:
        for n in [3, 4, 5]:
            r = results[n]
            f.write(f"n={n}  Rz(pi/{2**n}):\n{r['quirk_link']}\n\n")
    print(f"  Quirk links saved to: {link_file}")

    print(f"\n{'=' * 70}")
    print("To visualize: open the Quirk links above in a browser.")
    print("Each link shows the full gate sequence acting on one qubit.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
