"""
Team Synthesis Part 1: Learn the language of Clifford+T
=========================================================
Demonstrates H, S, T, and CNOT gates on simple input states
using Bloqade Squin and PyQrack.
"""

import numpy as np
from bloqade import squin
from bloqade.pyqrack import StackMemorySimulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_dm(kernel, n_qubits):
    """Run a kernel once and return the reduced density matrix."""
    sim = StackMemorySimulator(min_qubits=n_qubits)
    task = sim.task(kernel)
    result = task.run()
    return StackMemorySimulator.reduced_density_matrix(result)


def fmt_state(rho):
    """Extract the state vector from a pure-state density matrix."""
    eigvals, eigvecs = np.linalg.eigh(rho)
    idx = np.argmax(eigvals)
    psi = eigvecs[:, idx]
    # Fix global phase: make first nonzero entry real and positive
    for i in range(len(psi)):
        if abs(psi[i]) > 1e-10:
            psi = psi * np.exp(-1j * np.angle(psi[i]))
            break
    return psi


def fmt_complex(z):
    """Format a complex number nicely."""
    if abs(z.imag) < 1e-10:
        return f"{z.real:.4f}"
    elif abs(z.real) < 1e-10:
        return f"{z.imag:.4f}i"
    else:
        sign = "+" if z.imag >= 0 else "-"
        return f"{z.real:.4f}{sign}{abs(z.imag):.4f}i"


# ===========================================================================
# HADAMARD GATE
# ===========================================================================

@squin.kernel
def h_on_zero():
    q = squin.qalloc(1)
    squin.h(q[0])
    return q


@squin.kernel
def h_on_one():
    q = squin.qalloc(1)
    squin.x(q[0])
    squin.h(q[0])
    return q


@squin.kernel
def h_on_plus():
    q = squin.qalloc(1)
    squin.h(q[0])
    squin.h(q[0])
    return q


# ===========================================================================
# S GATE
# ===========================================================================

@squin.kernel
def s_on_zero():
    q = squin.qalloc(1)
    squin.s(q[0])
    return q


@squin.kernel
def s_on_one():
    q = squin.qalloc(1)
    squin.x(q[0])
    squin.s(q[0])
    return q


@squin.kernel
def s_on_plus():
    q = squin.qalloc(1)
    squin.h(q[0])
    squin.s(q[0])
    return q


# ===========================================================================
# T GATE
# ===========================================================================

@squin.kernel
def t_on_zero():
    q = squin.qalloc(1)
    squin.t(q[0])
    return q


@squin.kernel
def t_on_one():
    q = squin.qalloc(1)
    squin.x(q[0])
    squin.t(q[0])
    return q


@squin.kernel
def t_on_plus():
    q = squin.qalloc(1)
    squin.h(q[0])
    squin.t(q[0])
    return q


# ===========================================================================
# CNOT GATE
# ===========================================================================

@squin.kernel
def cnot_00():
    q = squin.qalloc(2)
    squin.cx(q[0], q[1])
    return q


@squin.kernel
def cnot_10():
    q = squin.qalloc(2)
    squin.x(q[0])
    squin.cx(q[0], q[1])
    return q


@squin.kernel
def cnot_01():
    q = squin.qalloc(2)
    squin.x(q[1])
    squin.cx(q[0], q[1])
    return q


@squin.kernel
def cnot_11():
    q = squin.qalloc(2)
    squin.x(q[0])
    squin.x(q[1])
    squin.cx(q[0], q[1])
    return q


@squin.kernel
def bell_state():
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.cx(q[0], q[1])
    return q


# ===========================================================================
# KEY RELATIONSHIPS: T² = S, S² = Z
# ===========================================================================

@squin.kernel
def t_squared_on_plus():
    q = squin.qalloc(1)
    squin.h(q[0])
    squin.t(q[0])
    squin.t(q[0])
    return q


@squin.kernel
def s_on_plus_direct():
    q = squin.qalloc(1)
    squin.h(q[0])
    squin.s(q[0])
    return q


@squin.kernel
def s_squared_on_plus():
    q = squin.qalloc(1)
    squin.h(q[0])
    squin.s(q[0])
    squin.s(q[0])
    return q


@squin.kernel
def z_on_plus_direct():
    q = squin.qalloc(1)
    squin.h(q[0])
    squin.z(q[0])
    return q


# ===========================================================================
# RUN ALL DEMONSTRATIONS
# ===========================================================================

def main():
    labels_2q = ["00", "01", "10", "11"]

    print("=" * 60)
    print("HADAMARD (H) GATE")
    print("Matrix: (1/√2) [[1, 1], [1, -1]]")
    print("=" * 60)

    rho = get_dm(h_on_zero, 1)
    psi = fmt_state(rho)
    print(f"H|0⟩ = {fmt_complex(psi[0])}|0⟩ + {fmt_complex(psi[1])}|1⟩")
    print(f"     = |+⟩ = (|0⟩ + |1⟩)/√2")

    rho = get_dm(h_on_one, 1)
    psi = fmt_state(rho)
    print(f"H|1⟩ = {fmt_complex(psi[0])}|0⟩ + {fmt_complex(psi[1])}|1⟩")
    print(f"     = |−⟩ = (|0⟩ − |1⟩)/√2")

    rho = get_dm(h_on_plus, 1)
    psi = fmt_state(rho)
    print(f"HH|0⟩ = {fmt_complex(psi[0])}|0⟩ + {fmt_complex(psi[1])}|1⟩")
    print(f"      = |0⟩  (H is its own inverse: H² = I)")

    # ---
    print("\n" + "=" * 60)
    print("S (PHASE) GATE")
    print("Matrix: [[1, 0], [0, i]]")
    print("=" * 60)

    rho = get_dm(s_on_zero, 1)
    psi = fmt_state(rho)
    print(f"S|0⟩ = {fmt_complex(psi[0])}|0⟩ + {fmt_complex(psi[1])}|1⟩")
    print(f"     = |0⟩  (S leaves |0⟩ unchanged)")

    rho = get_dm(s_on_one, 1)
    psi = fmt_state(rho)
    print(f"S|1⟩ = {fmt_complex(psi[0])}|0⟩ + {fmt_complex(psi[1])}|1⟩")
    print(f"     = i|1⟩  (applies phase i to |1⟩)")

    rho = get_dm(s_on_plus, 1)
    psi = fmt_state(rho)
    print(f"S|+⟩ = {fmt_complex(psi[0])}|0⟩ + {fmt_complex(psi[1])}|1⟩")
    print(f"     = (|0⟩ + i|1⟩)/√2  (rotates to Y-axis: |i⟩)")

    # ---
    print("\n" + "=" * 60)
    print("T GATE")
    print("Matrix: [[1, 0], [0, e^(iπ/4)]]")
    print("=" * 60)

    rho = get_dm(t_on_zero, 1)
    psi = fmt_state(rho)
    print(f"T|0⟩ = {fmt_complex(psi[0])}|0⟩ + {fmt_complex(psi[1])}|1⟩")
    print(f"     = |0⟩  (T leaves |0⟩ unchanged)")

    rho = get_dm(t_on_one, 1)
    psi = fmt_state(rho)
    print(f"T|1⟩ = {fmt_complex(psi[0])}|0⟩ + {fmt_complex(psi[1])}|1⟩")
    print(f"     = e^(iπ/4)|1⟩  (π/4 phase on |1⟩)")

    rho = get_dm(t_on_plus, 1)
    psi = fmt_state(rho)
    print(f"T|+⟩ = {fmt_complex(psi[0])}|0⟩ + {fmt_complex(psi[1])}|1⟩")
    print(f"     = (|0⟩ + e^(iπ/4)|1⟩)/√2  (π/8 rotation on Bloch sphere)")

    # ---
    print("\n" + "=" * 60)
    print("CNOT (CX) GATE")
    print("Matrix: |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩")
    print("=" * 60)

    for name, kernel, desc in [
        ("CNOT|00⟩", cnot_00, "|00⟩  (control=0 → target unchanged)"),
        ("CNOT|10⟩", cnot_10, "|11⟩  (control=1 → target flipped)"),
        ("CNOT|01⟩", cnot_01, "|01⟩  (control=0 → target unchanged)"),
        ("CNOT|11⟩", cnot_11, "|10⟩  (control=1 → target flipped)"),
    ]:
        rho = get_dm(kernel, 2)
        psi = fmt_state(rho)
        terms = [f"{fmt_complex(psi[i])}|{labels_2q[i]}⟩"
                 for i in range(4) if abs(psi[i]) > 1e-10]
        print(f"{name} = {' + '.join(terms)}")
        print(f"       = {desc}")

    # Bell state
    rho = get_dm(bell_state, 2)
    psi = fmt_state(rho)
    terms = [f"{fmt_complex(psi[i])}|{labels_2q[i]}⟩"
             for i in range(4) if abs(psi[i]) > 1e-10]
    print(f"\nCNOT(H⊗I)|00⟩ = {' + '.join(terms)}")
    print(f"               = (|00⟩ + |11⟩)/√2 = |Φ+⟩  (Bell state)")

    # ---
    print("\n" + "=" * 60)
    print("KEY RELATIONSHIPS")
    print("=" * 60)

    rho_tt = get_dm(t_squared_on_plus, 1)
    rho_s = get_dm(s_on_plus_direct, 1)
    overlap = float(np.real(np.trace(rho_tt @ rho_s)))
    print(f"T² = S ?  Overlap of T²|+⟩ and S|+⟩: {overlap:.6f}")

    rho_ss = get_dm(s_squared_on_plus, 1)
    rho_z = get_dm(z_on_plus_direct, 1)
    overlap = float(np.real(np.trace(rho_ss @ rho_z)))
    print(f"S² = Z ?  Overlap of S²|+⟩ and Z|+⟩: {overlap:.6f}")
    print("Hierarchy: T → S = T² → Z = S² = T⁴")

    print("\n✓ All simulations completed successfully.")


if __name__ == "__main__":
    main()
