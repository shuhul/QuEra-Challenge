# Synthesis Code

All code for the Team Synthesis track of the QuEra Challenge.

## Directory Structure

```
synthesis/
├── part1_clifford_t/
│   └── simulate_gates.py        # H, S, T, CNOT on simple states (Bloqade)
├── part2_gate_synthesis/
│   ├── exact_decompositions.py   # Exact Rz for n=0,1,2 with Bloch animations
│   └── approximate_synthesis.py  # SK (from scratch) + RS (pygridsynth) for n=3,4,5
├── part3_magic_injection/
│   └── magic_state_injection.py  # T-gate injection protocol, benchmarks Rz family
├── part4_steane_code/
│   ├── logical_qubit.py          # Steane [[7,1,3]] encoding, transversal gates, logical T
│   └── verify_magic_state.py     # Standalone magic state verification
└── common/
    └── generate_figures.py       # Bloch sphere figures (H, S, T, continuous)
```

## Running

All scripts are standalone and can be run from the project root:

```bash
uv sync
uv run python synthesis/part1_clifford_t/simulate_gates.py
uv run python synthesis/part2_gate_synthesis/exact_decompositions.py
uv run python synthesis/part2_gate_synthesis/approximate_synthesis.py
uv run python synthesis/part3_magic_injection/magic_state_injection.py
uv run python synthesis/part4_steane_code/logical_qubit.py
uv run python synthesis/common/generate_figures.py
```

## Expected Output

### Part 1: `simulate_gates.py`
Prints gate action verification for all Clifford+T gates on simple states:
- H|0> = |+>, H|1> = |->, HH|0> = |0>
- S|0> = |0>, S|1> = i|1>, S|+> = |i>
- T|0> = |0>, T|1> = e^(ipi/4)|1>, T|+> = (|0> + e^(ipi/4)|1>)/sqrt(2)
- CNOT truth table, Bell state creation
- Verifies T^2 = S and S^2 = Z

### Part 2: `exact_decompositions.py`
- Exact Clifford+T decompositions for Rz(pi), Rz(pi/2), Rz(pi/4)
- Bloch sphere animations saved to `figures/`

### Part 2: `approximate_synthesis.py`
- Solovay-Kitaev algorithm (depths 0-4) for n=3,4,5
- Ross-Selinger gridsynth at multiple epsilon values
- Comparison table and plots saved to `figures/`
- Convergence GIFs saved to `figures/`

### Part 3: `magic_state_injection.py`
- Verifies injection protocol (overlap = 1.0 for all test states)
- Benchmarks full Rz(pi/2^n) family via injection
- Prints cost analysis (ancilla count, CNOT count, measurements)

### Part 4: `logical_qubit.py`
- Steane code encoding verification (logical |0>, |+>, S|+>)
- Logical T injection on 14 qubits
- Full cost comparison: physical vs logical qubit overhead

### `generate_figures.py`
Generates Bloch sphere figures:
- `figures/bloch_gates_HST.{png,pdf}` — T, S, H gate visualizations
- `figures/bloch_continuous.{png,pdf}` — Continuous vs discrete Bloch sphere
- `figures/cnot_truth_table.{png,pdf}` — CNOT truth table
