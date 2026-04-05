# STAR Part 3: Teleport a Non-Clifford Rotation

**Author:** Joshua

Constructs a gate-teleportation protocol that injects an arbitrary Rz(θ_L)
rotation into a STAR d=3 logical qubit using an ancillary logical patch and
post-selection (Repeat-Until-Success, since Tsim does not support feed-forward).

## Protocol

1. Two STAR d=3 logical qubits (data patch qubits 0–8, ancilla patch qubits 17–25)
2. Both initialized in |+⟩_L via transversal RX
3. Ancilla left column (q17, q20, q23) gets transversal Rz(θ_phys)
4. Transversal CNOT: data (control) → ancilla (target)
5. Ancilla data qubits measured in X basis
6. Post-select on ancilla logical X = 0 (XOR of first-row MX outcomes)

## Files

- `part3.ipynb` — Full implementation: physical angle mapping, circuit construction, angle sweep, and raw measurement inspection

## Expected Output

- Verifies post-selected error rate matches sin²(θ_L π/2)
- Survival rate tracks cos²(θ_L π/2)
- Angle sweep from 0.001π to 0.5π with 50k shots per angle
- `teleportation_results.png` — Error rate and survival plots

## Writeup

- `writeup/STAR_part3_writeup.tex` — Full LaTeX writeup with protocol, qubit layout, cost analysis
