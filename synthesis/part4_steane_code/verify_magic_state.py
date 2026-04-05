import numpy as np
from bloqade import squin
from bloqade.pyqrack import StackMemorySimulator
from bloqade.types import Qubit
from kirin.dialects.ilist import IList
from typing import Any
from functools import reduce

Register = IList[Qubit, Any]


# --- Steane code encoding (logical |0>) ---
@squin.kernel
def encode_logical_zero() -> Register:
    q0 = squin.qalloc(7)
    squin.broadcast.h(q0)
    squin.cz(q0[0], q0[6])
    squin.cz(q0[1], q0[3])
    squin.cz(q0[4], q0[5])
    squin.cz(q0[0], q0[4])
    squin.cz(q0[5], q0[6])
    squin.cz(q0[1], q0[2])
    squin.cz(q0[0], q0[2])
    squin.cz(q0[3], q0[5])
    squin.h(q0[2])
    squin.h(q0[3])
    squin.h(q0[4])
    squin.h(q0[6])
    return q0


# --- Logical |1> = X on all qubits of logical |0> ---
@squin.kernel
def encode_logical_one() -> Register:
    q0 = encode_logical_zero()
    squin.broadcast.x(q0)
    return q0


# --- Magic state: T_L H_L |0>_L ---
@squin.kernel
def magic_state() -> Register:
    q0 = encode_logical_zero()
    squin.broadcast.h(q0)
    # Compute parity onto q0[6]
    squin.cx(q0[0], q0[6])
    squin.cx(q0[1], q0[6])
    squin.cx(q0[2], q0[6])
    squin.cx(q0[3], q0[6])
    squin.cx(q0[4], q0[6])
    squin.cx(q0[5], q0[6])
    # Apply T to parity qubit
    squin.t(q0[6])
    # Uncompute parity
    squin.cx(q0[5], q0[6])
    squin.cx(q0[4], q0[6])
    squin.cx(q0[3], q0[6])
    squin.cx(q0[2], q0[6])
    squin.cx(q0[1], q0[6])
    squin.cx(q0[0], q0[6])
    return q0


def get_dm(kernel, n_qubits):
    sim = StackMemorySimulator(min_qubits=n_qubits)
    task = sim.task(kernel)
    result = task.run()
    return StackMemorySimulator.reduced_density_matrix(result)


def main():
    print("=" * 60)
    print("MAGIC STATE VERIFICATION")
    print("=" * 60)

    # Get density matrices for logical |0>, |1>, and magic state
    rho_zero = get_dm(encode_logical_zero, 7)
    rho_one = get_dm(encode_logical_one, 7)
    rho_magic = get_dm(magic_state, 7)

    print(f"  rho_zero  shape: {rho_zero.shape}")
    print(f"  rho_one   shape: {rho_one.shape}")
    print(f"  rho_magic shape: {rho_magic.shape}")

    # Extract state vectors from pure-state density matrices
    # For a pure state rho = |psi><psi|, the eigenvector with eigenvalue 1 is |psi>
    evals_0, evecs_0 = np.linalg.eigh(rho_zero)
    psi_0_L = evecs_0[:, np.argmax(evals_0)]  # logical |0>

    evals_1, evecs_1 = np.linalg.eigh(rho_one)
    psi_1_L = evecs_1[:, np.argmax(evals_1)]  # logical |1>

    evals_m, evecs_m = np.linalg.eigh(rho_magic)
    psi_magic = evecs_m[:, np.argmax(evals_m)]  # magic state from circuit

    # Verify logical |0> and |1> are orthogonal
    overlap_01 = np.abs(np.dot(psi_0_L.conj(), psi_1_L))
    print(f"\n  <0_L|1_L> = {overlap_01:.6f}  (should be ~0)")

    # Construct expected magic state:
    #   |A> = 1/sqrt(2) * (|0>_L + e^(i*pi/4) |1>_L)
    phase = np.exp(1j * np.pi / 4)
    psi_expected = (psi_0_L + phase * psi_1_L) / np.sqrt(2)

    # Compute fidelity: |<expected|magic>|^2
    # Need to handle global phase: fidelity = |<a|b>|^2
    fidelity = np.abs(np.dot(psi_expected.conj(), psi_magic)) ** 2
    print(f"  Fidelity |<expected|magic>|^2 = {fidelity:.6f}")

    # Also check via density matrix overlap: Tr(rho_expected @ rho_magic)
    rho_expected = np.outer(psi_expected, psi_expected.conj())
    dm_overlap = np.real(np.trace(rho_expected @ rho_magic))
    print(f"  Density matrix overlap         = {dm_overlap:.6f}")

    # Check Bloch vector on logical subspace
    I1 = np.eye(2, dtype=complex)
    X1 = np.array([[0, 1], [1, 0]], dtype=complex)
    Y1 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z1 = np.diag([1.0, -1.0]).astype(complex)
    n = 7

    def build_logical(pauli):
        ops = [I1] * n
        ops[0] = pauli; ops[1] = pauli; ops[2] = pauli
        return reduce(np.kron, ops)

    X_L, Y_L, Z_L = build_logical(X1), build_logical(Y1), build_logical(Z1)

    xl = np.real(np.trace(rho_magic @ X_L))
    yl = np.real(np.trace(rho_magic @ Y_L))
    zl = np.real(np.trace(rho_magic @ Z_L))

    # Check both T|+> and T†|+> conventions
    # T|+>:  Bloch = (cos(pi/4), +sin(pi/4), 0)  =>  e^(+i*pi/4)
    # T†|+>: Bloch = (cos(pi/4), -sin(pi/4), 0)  =>  e^(-i*pi/4)
    ex_t = (np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0)
    ex_td = (np.cos(np.pi / 4), -np.sin(np.pi / 4), 0.0)

    print(f"\n  Logical Bloch vector:")
    print(f"    got:            X={xl:+.4f}  Y={yl:+.4f}  Z={zl:+.4f}")
    print(f"    T|+>  expected: X={ex_t[0]:+.4f}  Y={ex_t[1]:+.4f}  Z={ex_t[2]:+.4f}")
    print(f"    T†|+> expected: X={ex_td[0]:+.4f}  Y={ex_td[1]:+.4f}  Z={ex_td[2]:+.4f}")

    match_t = np.allclose([xl, yl, zl], ex_t, atol=1e-4)
    match_td = np.allclose([xl, yl, zl], ex_td, atol=1e-4)

    if match_t:
        print(f"\n  PASS: magic state = 1/sqrt(2) (|0>_L + e^(+i*pi/4)|1>_L)")
    elif match_td:
        print(f"\n  PASS (T-dagger): magic state = 1/sqrt(2) (|0>_L + e^(-i*pi/4)|1>_L)")
        print(f"  Note: circuit applies T† not T in logical space")
    else:
        print(f"\n  FAIL: Bloch vector doesn't match either convention")
    print("=" * 60)


if __name__ == "__main__":
    main()
