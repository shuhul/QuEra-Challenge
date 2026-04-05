# STAR Code

All code for the Team STAR track of the QuEra Challenge.

## Directory Structure

```
star/
├── part1_surface_code/                          # Emiliano
│   ├── surface_code_d3.py                       # Distance-3 surface code + syndrome extraction
│   ├── validate_surface_code.py                 # Validation of surface code implementation
│   ├── STAR_P1_CNOT_IMPORTANT_SCHEDULE.png      # CNOT scheduling reference
│   ├── STAR_P1_Patch_geometry_IMPORTANT.png     # Patch geometry reference
│   ├── star_scaling_analysis.png                # Scaling analysis plot
│   ├── star_scaling_normalized.png              # Normalized scaling plot
│   └── star_scaling_plots/                      # Additional scaling analysis
│       ├── combined_summary.png
│       ├── depth_scaling.png
│       ├── detectors_measurements.png
│       ├── gate_counts.png
│       ├── qubit_scaling.png
│       └── star_metrics.csv
├── part2_star_fidelity/                         # Joshua
│   ├── star_fidelity_simulation.ipynb           # STAR fidelity plot reproduction (Tsim)
│   └── STARpart2.png                            # Fidelity plot output
├── part3_teleported_rotation/                   # Joshua
│   ├── part3.ipynb                              # Gate teleportation protocol (Tsim)
│   └── README.md                                # Protocol description
└── part4_qft_circuit/                           # Emiliano
    └── qft_teleported_rotations.ipynb           # 4-qubit QFT with teleported non-Clifford gates
```

## Running

### Part 1: Surface Code
```bash
uv run python star/part1_surface_code/surface_code_d3.py
uv run python star/part1_surface_code/validate_surface_code.py
```

### Part 2: STAR Fidelity (Jupyter notebook)
```bash
uv run jupyter lab star/part2_star_fidelity/star_fidelity_simulation.ipynb
```

### Part 3: Teleported Rotation (Jupyter notebook)
```bash
uv run jupyter lab star/part3_teleported_rotation/part3.ipynb
```

### Part 4: QFT Circuit (Jupyter notebook)
```bash
uv run jupyter lab star/part4_qft_circuit/qft_teleported_rotations.ipynb
```

## Expected Output

### Part 1: `surface_code_d3.py`
- Constructs a distance-3 surface code in Tsim
- Simulates two rounds of syndrome extraction
- Identifies data/ancilla qubits and stabilizer checks
- Outputs scaling analysis (plots in `star_scaling_plots/`)

### Part 2: `star_fidelity_simulation.ipynb`
- Loads STAR circuits from challenge assets
- Simulates with noise using Tsim
- Reproduces the fidelity plot from the challenge description
- Output: `STARpart2.png`

### Part 3: `part3.ipynb`
- Builds STAR d=3 gate-teleportation circuit for arbitrary Rz(θ_L)
- Maps logical angle to physical angle via `physical_angle()` helper
- Sweeps θ_L from 0.001π to 0.5π, verifies error rate = sin²(θ_L π/2)
- Post-selection survival tracks cos²(θ_L π/2)
- Output: `teleportation_results.png`

### Part 4: `qft_teleported_rotations.ipynb`
- Constructs a 4-qubit QFT circuit
- All non-Clifford gates are teleported in (not applied directly)
- Evaluates scalability of the approach
