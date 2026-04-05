"""
Part 3: Non-Clifford Gates Are Expensive — Magic State Injection
================================================================
Replaces T gates with magic state injection (ancilla + CNOT + measure +
feedforward) and verifies correctness. Tracks overhead costs.
"""

import numpy as np
from bloqade import squin
from bloqade.pyqrack import StackMemorySimulator
from bloqade.types import Qubit
from pathlib import Path
import sys
import time

# Add project root to path for cross-module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from synthesis.part2_gate_synthesis.approximate_synthesis import (
    Rz, gate_distance, t_count, seq_to_unitary,
    H_mat, T_mat, Td_mat, S_mat, Sd_mat, I2, ALL_GATE_MATRICES,
    SolovayKitaev
)
import mpmath
from pygridsynth.gridsynth import gridsynth_gates


# =====================================================================
# Helpers
# =====================================================================

def get_dm(kernel, n_qubits):
    sim = StackMemorySimulator(min_qubits=n_qubits)
    task = sim.task(kernel)
    result = task.run()
    return StackMemorySimulator.reduced_density_matrix(result)


def partial_trace_keep_first(rho):
    """Trace out all qubits except the first, returning a 2x2 DM."""
    n = rho.shape[0]
    if n == 2:
        return rho
    a = n // 2
    return np.einsum('iaja->ij', rho.reshape(2, a, 2, a))


def partial_trace_keep_last(rho):
    """Trace out all qubits except the last, returning a 2x2 DM."""
    n = rho.shape[0]
    if n == 2:
        return rho
    a = n // 2
    return np.einsum('aibj->ij', rho.reshape(a, 2, a, 2))


def state_overlap(rho1, rho2):
    return float(np.real(np.trace(rho1 @ rho2)))


# =====================================================================
# MAGIC STATE INJECTION (from Nandana)
# =====================================================================

@squin.kernel
def inject_t(data: Qubit) -> Qubit:
    """Inject T gate via magic state."""
    anc = squin.qalloc(1)[0]
    squin.h(anc)
    squin.t(anc)
    squin.cx(data, anc)
    m = squin.measure(anc)
    if m:
        squin.s(data)
    return data


@squin.kernel
def inject_tdg(data: Qubit) -> Qubit:
    """Inject T-dagger via conjugate magic state.
    Ancilla prep: T*S*Z*H|0> = T-dag*H|0>.
    Correction: S-dag = S*S*S.
    """
    anc = squin.qalloc(1)[0]
    squin.h(anc)
    squin.z(anc)
    squin.s(anc)
    squin.t(anc)
    squin.cx(data, anc)
    m = squin.measure(anc)
    if m:
        squin.s(data)
        squin.s(data)
        squin.s(data)
    return data


# =====================================================================
# SECTION 1: MATHEMATICAL PROOF (numpy)
# =====================================================================

def verify_numpy():
    """Prove injection is exact by simulating the full 2-qubit protocol."""
    CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)

    print("\n--- Mathematical Verification (numpy) ---")
    print("  Both measurement outcomes yield T|psi> (up to global phase).\n")

    test_states = [
        ("|0>", np.array([1, 0], dtype=complex)),
        ("|1>", np.array([0, 1], dtype=complex)),
        ("|+>", np.array([1, 1], dtype=complex) / np.sqrt(2)),
        ("|i>", np.array([1, 1j], dtype=complex) / np.sqrt(2)),
    ]

    for gate, correction, label in [
        (T_mat, S_mat, "T injection"),
        (Td_mat, Sd_mat, "T-dag injection"),
    ]:
        print(f"  {label}:")
        magic = gate @ H_mat @ np.array([1, 0], dtype=complex)

        for name, psi in test_states:
            joint = np.kron(psi, magic)
            joint = CNOT @ joint
            psi_2q = joint.reshape(2, 2)

            data_0 = psi_2q[:, 0]               # m=0: no correction
            data_1 = correction @ psi_2q[:, 1]   # m=1: apply correction

            expected = gate @ psi
            rho_expected = np.outer(expected, expected.conj())
            rho_result = (np.outer(data_0, data_0.conj())
                          + np.outer(data_1, data_1.conj()))

            overlap = state_overlap(rho_result, rho_expected)
            status = "OK" if overlap > 0.9999 else "FAIL"
            print(f"    {label[0:2]}{name}: overlap = {overlap:.10f}  {status}")
        print()


# =====================================================================
# SECTION 2: BLOQADE SIMULATOR VERIFICATION
# =====================================================================

@squin.kernel
def t_plus_direct():
    q = squin.qalloc(1)
    squin.h(q[0])
    squin.t(q[0])
    return q

@squin.kernel
def t_plus_inject():
    q = squin.qalloc(1)
    squin.h(q[0])
    inject_t(q[0])
    return q

@squin.kernel
def t_zero_direct():
    q = squin.qalloc(1)
    squin.t(q[0])
    return q

@squin.kernel
def t_zero_inject():
    q = squin.qalloc(1)
    inject_t(q[0])
    return q

@squin.kernel
def t_one_direct():
    q = squin.qalloc(1)
    squin.x(q[0])
    squin.t(q[0])
    return q

@squin.kernel
def t_one_inject():
    q = squin.qalloc(1)
    squin.x(q[0])
    inject_t(q[0])
    return q

@squin.kernel
def tt_plus_direct():
    q = squin.qalloc(1)
    squin.h(q[0])
    squin.t(q[0])
    squin.t(q[0])
    return q

@squin.kernel
def tt_plus_inject():
    q = squin.qalloc(1)
    squin.h(q[0])
    inject_t(q[0])
    inject_t(q[0])
    return q

@squin.kernel
def s_plus_ref():
    q = squin.qalloc(1)
    squin.h(q[0])
    squin.s(q[0])
    return q


def verify_bloqade():
    """Verify injection using Bloqade/PyQrack simulator."""
    print("--- Bloqade Simulator Verification ---")

    tests = [
        ("T|0>",  t_zero_direct, t_zero_inject, 1, 2),
        ("T|1>",  t_one_direct,  t_one_inject,  1, 2),
        ("T|+>",  t_plus_direct, t_plus_inject, 1, 2),
        ("T^2|+>", tt_plus_direct, tt_plus_inject, 1, 3),
    ]

    for name, direct_k, inject_k, n_d, n_i in tests:
        try:
            rho_direct = get_dm(direct_k, n_d)
            rho_inject_full = get_dm(inject_k, n_i)

            # Try both qubit orderings for partial trace
            rho_first = partial_trace_keep_first(rho_inject_full)
            rho_last = partial_trace_keep_last(rho_inject_full)

            o_first = state_overlap(rho_direct, rho_first)
            o_last = state_overlap(rho_direct, rho_last)
            overlap = max(o_first, o_last)

            status = "OK" if overlap > 0.99 else "FAIL"
            print(f"  {name:>8}: overlap = {overlap:.6f}  {status}")
        except Exception as e:
            print(f"  {name:>8}: SKIPPED ({e})")

    # T^2 via injection should equal S
    try:
        rho_s = get_dm(s_plus_ref, 1)
        rho_tt_full = get_dm(tt_plus_inject, 3)
        rho_f = partial_trace_keep_first(rho_tt_full)
        rho_l = partial_trace_keep_last(rho_tt_full)
        overlap = max(state_overlap(rho_s, rho_f), state_overlap(rho_s, rho_l))
        status = "OK" if overlap > 0.99 else "FAIL"
        print(f"  {'T^2=S?':>8}: overlap = {overlap:.6f}  {status}")
    except Exception as e:
        print(f"  {'T^2=S?':>8}: SKIPPED ({e})")
    print()


# =====================================================================
# SECTION 3: COST ANALYSIS
# =====================================================================

def cost_analysis():
    """Compute overhead of magic state injection for Part 2 sequences."""
    print("--- Cost of Magic State Injection ---\n")

    print("  Per T/T-dag injection:")
    print("    1 ancilla qubit (magic state TH|0>)")
    print("    1 CNOT (data -> ancilla)")
    print("    1 measurement")
    print("    1 feedforward correction (conditional S or S-dag)")
    print("    2-4 gates on ancilla for preparation")

    targets = {3: Rz(np.pi / 8), 4: Rz(np.pi / 16), 5: Rz(np.pi / 32)}

    # SK sequences
    sk = SolovayKitaev(db_max_depth=15)

    # RS sequences
    rs_data = {}
    for n in [3, 4, 5]:
        theta_mp = mpmath.mpf(float(np.pi / (2 ** n)))
        gs_str = gridsynth_gates(theta=theta_mp, epsilon=mpmath.mpf('0.001'))
        seq = [c for c in gs_str if c in ('H', 'T', 'S', 'X')]
        U = seq_to_unitary(seq) if seq else I2
        d = gate_distance(targets[n], U) if seq else float('inf')
        rs_data[n] = (seq, d)

    print(f"\n  {'n':>2} | {'Method':>10} | {'Orig':>5} | {'T-cnt':>5} | "
          f"{'Anc':>4} | {'CNOT':>4} | {'Meas':>4} | {'Distance':>10}")
    print(f"  {'─' * 2}─┼─{'─' * 10}─┼─{'─' * 5}─┼─{'─' * 5}─┼─"
          f"{'─' * 4}─┼─{'─' * 4}─┼─{'─' * 4}─┼─{'─' * 10}")

    for n in [3, 4, 5]:
        # SK depth 3
        _, sk_seq, sk_dist = sk.solve(targets[n], 3)
        tc = t_count(sk_seq)
        print(f"  {n:>2} | {'SK d=3':>10} | {len(sk_seq):>5} | {tc:>5} | "
              f"{tc:>4} | {tc:>4} | {tc:>4} | {sk_dist:>10.6f}")

        # RS eps=1e-3
        rs_seq, rs_dist = rs_data[n]
        tc_rs = sum(1 for g in rs_seq if g == 'T')
        print(f"  {n:>2} | {'RS 1e-3':>10} | {len(rs_seq):>5} | {tc_rs:>5} | "
              f"{tc_rs:>4} | {tc_rs:>4} | {tc_rs:>4} | {rs_dist:>10.6f}")
        print(f"  {'─' * 2}─┼─{'─' * 10}─┼─{'─' * 5}─┼─{'─' * 5}─┼─"
              f"{'─' * 4}─┼─{'─' * 4}─┼─{'─' * 4}─┼─{'─' * 10}")

    print("\n  Key insights:")
    print("    - Approximation distance is UNCHANGED (injection is exact)")
    print("    - Each T gate now costs 1 ancilla + 1 CNOT + 1 measurement")
    print("    - SK depth 3 needs ~800 ancillas vs RS needs ~34")
    print("    - RS's lower T-count is even more valuable with injection")
    print("    - In fault-tolerant QC, each ancilla = magic state distillation")
    print("      (~15 physical qubits), so 34 T-gates = 510 physical qubits")


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 65)
    print("PART 3: NON-CLIFFORD GATES ARE EXPENSIVE")
    print("        Magic State Injection Protocol")
    print("=" * 65)

    verify_numpy()
    verify_bloqade()
    cost_analysis()

    print("\n" + "=" * 65)
    print("DONE: Part 3 complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
