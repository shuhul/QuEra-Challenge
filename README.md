# Team Calbits — YQuantum 2026 QuEra Challenge

**Team:** [shuhul](https://github.com/shuhul), [emiIiano](https://github.com/emiIiano), [ThuwarageshJ](https://github.com/ThuwarageshJ), [nmadhu6002](https://github.com/nmadhu6002), [jujurover](https://github.com/jujurover)

**Write-up:** The final technical report for this challenge can be found in [`YQuantum_2026_Submission___Calbits.pdf`](./YQuantum_2026_Submission___Calbits.pdf).

**Presentation:** Calbits Presentation.pptx

### Repo Structure

```
├── synthesis/                                   # Gate Synthesis Track (Shuhul, Thuwa, Nandana)
│   ├── part1_clifford_t/
│   │   └── simulate_gates.py                    # H, S, T, CNOT verification with Bloqade
│   ├── part2_gate_synthesis/
│   │   ├── exact_decompositions.py              # Exact Rz decompositions (n=0,1,2)
│   │   ├── approximate_synthesis.py             # Solovay-Kitaev + Ross-Selinger (n=3,4,5)
│   │   └── rs_circuits.py                       # RS circuit diagrams + Quirk links
│   ├── part3_magic_injection/
│   │   └── magic_state_injection.py             # T-gate via magic state injection
│   ├── part4_steane_code/
│   │   ├── logical_qubit.py                     # Steane [[7,1,3]] code
│   │   └── verify_magic_state.py                # Magic state verification
│   └── common/
│       └── generate_figures.py                  # Bloch sphere figures
│
├── star/                                        # STAR Architecture Track (Joshua, Emiliano)
│   ├── part1_surface_code/
│   │   ├── surface_code_d3.py                   # Distance-3 surface code (Tsim)
│   │   ├── validate_surface_code.py             # Surface code validation
│   │   ├── noise_models.py                      # Circuit-level noise channels
│   │   ├── surface_code_generator.py            # Arbitrary-distance geometry
│   │   └── noise_models_multiscale.py           # Multi-distance noise analysis
│   ├── part2_star_fidelity/
│   │   └── star_fidelity_simulation.ipynb       # STAR fidelity reproduction
│   ├── part3_teleported_rotation/
│   │   └── part3.ipynb                          # Gate teleportation protocol (Tsim)
│   └── part4_qft_circuit/
│       ├── qft_resource_analysis.py             # QFT resource analysis with STAR
│       └── qft_teleported_rotations.ipynb       # 4-qubit QFT with teleported gates
│
├── figures/                                     # Generated plots and visualizations
├── outputs/                                     # RS gate sequences and Quirk links
├── assets/                                      # Challenge-provided reference material
└── pyproject.toml                               # Dependencies (managed with uv)
```

### Running

**Install dependencies:**
```bash
uv sync
```

**Gate Synthesis Track:**
```bash
# Part 1: Clifford+T gate verification
uv run python synthesis/part1_clifford_t/simulate_gates.py

# Part 2: Gate synthesis (exact + approximate)
uv run python synthesis/part2_gate_synthesis/exact_decompositions.py
uv run python synthesis/part2_gate_synthesis/approximate_synthesis.py
uv run python synthesis/part2_gate_synthesis/rs_circuits.py

# Part 3: Magic state injection
uv run python synthesis/part3_magic_injection/magic_state_injection.py

# Part 4: Steane code logical qubit
uv run python synthesis/part4_steane_code/logical_qubit.py

# Generate Bloch sphere figures
uv run python synthesis/common/generate_figures.py
```

**STAR Architecture Track:**
```bash
# Part 1: Surface code
uv run python star/part1_surface_code/surface_code_d3.py
uv run python star/part1_surface_code/validate_surface_code.py

# Part 2: STAR fidelity (Jupyter notebook)
uv run jupyter lab star/part2_star_fidelity/star_fidelity_simulation.ipynb

# Part 3: Teleported rotation (Jupyter notebook)
uv run jupyter lab star/part3_teleported_rotation/part3.ipynb

# Part 4: QFT resource analysis + circuit
uv run python star/part4_qft_circuit/qft_resource_analysis.py
uv run jupyter lab star/part4_qft_circuit/qft_teleported_rotations.ipynb
```

### AI Tools Disclosure

This project used Claude Code (Anthropic) for code assistance, debugging, and LaTeX generation. All team members understand the underlying quantum computing concepts, can explain the algorithms and results presented, and made the key design decisions.
