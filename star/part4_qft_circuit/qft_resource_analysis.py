"""
Part 4: 4-Qubit QFT with STAR-Teleported Non-Clifford Gates
===========================================================
Distance d=3 baseline only.

This module:
  - Defines the 4-qubit QFT gate structure
  - Identifies which gates require STAR teleportation
  - Parses star_d=3.stim using text parsing (Tsim-compatible)
  - Clearly separates DIRECT measurements from INFERRED quantities
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import re

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Default path — can be overridden
DEFAULT_STIM_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "star_circuits"

# ══════════════════════════════════════════════════════════════════════════════
# QFT GATE STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CliffordGate:
    """A Clifford gate in the QFT (no STAR needed)."""
    name: str
    qubits: Tuple[int, ...]
    angle: Optional[str] = None  # For CP gates
    
    def __repr__(self):
        if self.angle:
            return f"{self.name}({self.angle}) on {self.qubits}"
        return f"{self.name} on {self.qubits}"


@dataclass
class NonCliffordGate:
    """A non-Clifford gate requiring STAR teleportation."""
    name: str
    control: int
    target: int
    angle: str           # e.g., "pi/4"
    star_rotation: str   # The Rz angle injected by STAR, e.g., "pi/8"
    
    def __repr__(self):
        return f"{self.name}({self.angle}) on q{self.control}-q{self.target} → STAR injects Rz({self.star_rotation})"


@dataclass
class QFT4Structure:
    """Complete gate structure for 4-qubit QFT."""
    clifford_gates: List[CliffordGate] = field(default_factory=list)
    non_clifford_gates: List[NonCliffordGate] = field(default_factory=list)
    
    @property
    def total_gates(self) -> int:
        return len(self.clifford_gates) + len(self.non_clifford_gates)
    
    @property
    def star_gadgets_needed(self) -> int:
        return len(self.non_clifford_gates)


def build_qft4_structure() -> QFT4Structure:
    """
    Construct the 4-qubit QFT gate sequence.
    
    Standard QFT circuit (before bit-reversal swaps):
    
        q0: ─H─CP(π/2)─CP(π/4)─CP(π/8)─────────────────────────────
              │        │        │
        q1: ──●────────│────────│────H─CP(π/2)─CP(π/4)────────────
                       │        │        │        │
        q2: ───────────●────────│────────●────────│────H─CP(π/2)──
                                │                 │        │
        q3: ────────────────────●─────────────────●────────●────H─
    
    Plus SWAP(0,3) and SWAP(1,2) for bit reversal.
    
    Gate classification:
      - H: Clifford
      - CP(π/2) = controlled-S: Clifford
      - CP(π/4) = controlled-T: NON-Clifford → needs STAR
      - CP(π/8): NON-Clifford → needs STAR
      - SWAP: Clifford (3 CNOTs)
    
    CP(θ) decomposition for STAR:
        CP(θ) = CNOT · (I ⊗ Rz(θ/2)) · CNOT · (I ⊗ Rz(-θ/2))
        
    For CP(π/4): need Rz(±π/8) → STAR teleports π/8 rotation
    For CP(π/8): need Rz(±π/16) → STAR teleports π/16 rotation
    """
    
    qft = QFT4Structure()
    
    # ── Clifford gates ────────────────────────────────────────────────────────
    
    # Layer 1: q0
    qft.clifford_gates.append(CliffordGate("H", (0,)))
    qft.clifford_gates.append(CliffordGate("CP", (0, 1), "π/2"))
    # CP(π/4) on (0,2) is NON-Clifford — handled below
    # CP(π/8) on (0,3) is NON-Clifford — handled below
    
    # Layer 2: q1
    qft.clifford_gates.append(CliffordGate("H", (1,)))
    qft.clifford_gates.append(CliffordGate("CP", (1, 2), "π/2"))
    # CP(π/4) on (1,3) is NON-Clifford — handled below
    
    # Layer 3: q2
    qft.clifford_gates.append(CliffordGate("H", (2,)))
    qft.clifford_gates.append(CliffordGate("CP", (2, 3), "π/2"))
    
    # Layer 4: q3
    qft.clifford_gates.append(CliffordGate("H", (3,)))
    
    # Bit-reversal swaps
    qft.clifford_gates.append(CliffordGate("SWAP", (0, 3)))
    qft.clifford_gates.append(CliffordGate("SWAP", (1, 2)))
    
    # ── Non-Clifford gates (require STAR) ─────────────────────────────────────
    
    qft.non_clifford_gates.append(NonCliffordGate(
        name="CP",
        control=0,
        target=2,
        angle="π/4",
        star_rotation="π/8",
    ))
    
    qft.non_clifford_gates.append(NonCliffordGate(
        name="CP",
        control=0,
        target=3,
        angle="π/8",
        star_rotation="π/16",
    ))
    
    qft.non_clifford_gates.append(NonCliffordGate(
        name="CP",
        control=1,
        target=3,
        angle="π/4",
        star_rotation="π/8",
    ))
    
    return qft


# Global instance for easy import
QFT_4_GATES = build_qft4_structure()


# ══════════════════════════════════════════════════════════════════════════════
# STAR CIRCUIT PARSING (TEXT-BASED, TSIM-COMPATIBLE)
# ══════════════════════════════════════════════════════════════════════════════

# Instruction categories for counting
SINGLE_QUBIT_GATES = {
    "H", "X", "Y", "Z", "S", "S_DAG", "T", "T_DAG",
    "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG",
    "SQRT_Z", "SQRT_Z_DAG", "I",
}

TWO_QUBIT_GATES = {
    "CX", "CNOT", "CY", "CZ", "SWAP", "ISWAP", "ISWAP_DAG",
    "SQRT_XX", "SQRT_YY", "SQRT_ZZ", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ",
}

RESET_OPS = {"R", "RX", "RY", "RZ", "R_X", "R_Y", "R_Z"}

MEASURE_OPS = {"M", "MR", "MX", "MY", "MZ", "MRX", "MRY", "MRZ", "MPP"}

NOISE_OPS = {
    "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR",
    "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "HERALDED_PAULI_CHANNEL_1",
}

ANNOTATION_OPS = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"}


@dataclass
class STARCircuitStats:
    """Statistics parsed directly from a STAR .stim file."""
    distance: int
    filepath: Path
    
    # Qubit counts
    total_qubits: int
    data_qubits: int           # Inferred from geometry (d²)
    ancilla_qubits: int        # Inferred from geometry
    
    # Gate counts (DIRECT from file)
    h_gates: int
    s_gates: int               # S and S_DAG combined
    t_gates: int               # T and T_DAG combined
    cx_gates: int              # All 2-qubit gates
    r_resets: int              # All reset operations
    
    # Measurement counts (DIRECT from file)
    measurements: int          # All measurement operations
    detectors: int             # DETECTOR annotations
    observables: int           # OBSERVABLE_INCLUDE annotations
    
    # Timing (DIRECT from file)
    tick_count: int            # Number of TICK operations
    
    # Noise (DIRECT - count existing noise ops)
    depolarize1_count: int
    depolarize2_count: int
    x_error_count: int
    z_error_count: int
    
    # Additional tracking
    total_lines: int
    instruction_counts: Dict[str, int] = field(default_factory=dict)
    unknown_instructions: Set[str] = field(default_factory=set)


def parse_instruction_line(line: str) -> Tuple[Optional[str], List[str], Optional[float]]:
    """
    Parse a single line from a .stim file.
    
    Returns:
        (instruction_name, targets, parameter) or (None, [], None) for non-instruction lines
    """
    # Strip whitespace and comments
    line = line.strip()
    if not line or line.startswith("#"):
        return None, [], None
    
    # Remove inline comments
    if "#" in line:
        line = line[:line.index("#")].strip()
    
    if not line:
        return None, [], None
    
    # Parse instruction name
    parts = line.split()
    if not parts:
        return None, [], None
    
    instruction = parts[0].upper()
    
    # Check for parameter in parentheses: DEPOLARIZE1(0.001)
    param = None
    if "(" in instruction:
        match = re.match(r"([A-Z_0-9]+)\(([^)]+)\)", instruction)
        if match:
            instruction = match.group(1)
            try:
                param = float(match.group(2))
            except ValueError:
                param = None
    
    # Remaining parts are targets
    targets = parts[1:] if len(parts) > 1 else []
    
    return instruction, targets, param


def count_targets(targets: List[str]) -> int:
    """
    Count the number of qubit targets in a target list.
    Handles various formats: plain integers, rec[-1], etc.
    """
    count = 0
    for t in targets:
        # Skip record references like rec[-1]
        if t.startswith("rec["):
            continue
        # Skip combiners
        if t == "*":
            continue
        # Try to extract qubit number
        try:
            # Handle formats like "0", "q0", etc.
            cleaned = re.sub(r"[^\d]", "", t)
            if cleaned:
                count += 1
        except (ValueError, AttributeError):
            continue
    return count


def extract_qubits(targets: List[str]) -> Set[int]:
    """Extract qubit indices from target list."""
    qubits = set()
    for t in targets:
        if t.startswith("rec["):
            continue
        if t == "*":
            continue
        try:
            cleaned = re.sub(r"[^\d]", "", t)
            if cleaned:
                qubits.add(int(cleaned))
        except (ValueError, AttributeError):
            continue
    return qubits


def parse_star_circuit(d: int, stim_dir: Optional[Path] = None) -> STARCircuitStats:
    """
    Parse a STAR circuit file using text-based parsing.
    
    This approach handles non-standard instructions (like R_Z) that
    stim.Circuit.from_file() cannot parse.
    
    Args:
        d: Code distance (must be odd)
        stim_dir: Directory containing star_d={d}.stim files
        
    Returns:
        STARCircuitStats with all DIRECTLY measured quantities
    """
    if stim_dir is None:
        stim_dir = DEFAULT_STIM_DIR
    
    filepath = stim_dir / f"star_d={d}.stim"
    if not filepath.exists():
        raise FileNotFoundError(f"STAR circuit not found: {filepath}")
    
    # Read file as text
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    # Initialize counters
    instruction_counts: Dict[str, int] = {}
    unknown_instructions: Set[str] = set()
    qubits_used: Set[int] = set()
    
    h_gates = 0
    s_gates = 0
    t_gates = 0
    cx_gates = 0
    r_resets = 0
    measurements = 0
    detectors = 0
    observables = 0
    tick_count = 0
    depolarize1_count = 0
    depolarize2_count = 0
    x_error_count = 0
    z_error_count = 0
    
    for line in lines:
        instruction, targets, param = parse_instruction_line(line)
        
        if instruction is None:
            continue
        
        # Track instruction frequency
        instruction_counts[instruction] = instruction_counts.get(instruction, 0) + 1
        
        # Extract qubits
        qubits_used.update(extract_qubits(targets))
        
        # Count by category
        if instruction == "H":
            h_gates += count_targets(targets)
        
        elif instruction in ("S", "S_DAG"):
            s_gates += count_targets(targets)
        
        elif instruction in ("T", "T_DAG"):
            t_gates += count_targets(targets)
        
        elif instruction in TWO_QUBIT_GATES:
            # Two-qubit gates: count pairs
            cx_gates += count_targets(targets) // 2
        
        elif instruction in RESET_OPS:
            r_resets += count_targets(targets)
        
        elif instruction in MEASURE_OPS:
            if instruction == "MPP":
                # MPP measures Pauli products - count as 1 measurement per target group
                measurements += 1
            else:
                measurements += count_targets(targets)
        
        elif instruction == "DETECTOR":
            detectors += 1
        
        elif instruction == "OBSERVABLE_INCLUDE":
            observables += 1
        
        elif instruction == "TICK":
            tick_count += 1
        
        elif instruction == "DEPOLARIZE1":
            depolarize1_count += count_targets(targets)
        
        elif instruction == "DEPOLARIZE2":
            depolarize2_count += count_targets(targets) // 2
        
        elif instruction == "X_ERROR":
            x_error_count += count_targets(targets)
        
        elif instruction == "Z_ERROR":
            z_error_count += count_targets(targets)
        
        elif instruction in SINGLE_QUBIT_GATES:
            pass  # Counted individually above
        
        elif instruction in ANNOTATION_OPS:
            pass  # Already handled
        
        elif instruction in NOISE_OPS:
            pass  # Already handled
        
        elif instruction in ("REPEAT", "}", "{"):
            pass  # Control flow - handled separately
        
        else:
            # Unknown instruction - track but don't fail
            unknown_instructions.add(instruction)
    
    total_qubits = len(qubits_used) if qubits_used else 0
    
    # d=3 surface code geometry: d² data qubits
    data_qubits = d * d
    ancilla_qubits = max(0, total_qubits - data_qubits)
    
    return STARCircuitStats(
        distance=d,
        filepath=filepath,
        total_qubits=total_qubits,
        data_qubits=data_qubits,
        ancilla_qubits=ancilla_qubits,
        h_gates=h_gates,
        s_gates=s_gates,
        t_gates=t_gates,
        cx_gates=cx_gates,
        r_resets=r_resets,
        measurements=measurements,
        detectors=detectors,
        observables=observables,
        tick_count=tick_count,
        depolarize1_count=depolarize1_count,
        depolarize2_count=depolarize2_count,
        x_error_count=x_error_count,
        z_error_count=z_error_count,
        total_lines=len(lines),
        instruction_counts=instruction_counts,
        unknown_instructions=unknown_instructions,
    )


# ══════════════════════════════════════════════════════════════════════════════
# QFT RESOURCE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QFTResourceSummary:
    """Complete resource accounting for 4-qubit QFT at distance d."""
    
    # QFT structure
    qft_logical_qubits: int
    clifford_gate_count: int
    non_clifford_gate_count: int
    star_gadgets_needed: int
    
    # Per-gadget resources (from STAR circuit)
    star_stats: STARCircuitStats
    
    # Total physical resources
    total_physical_qubits: int
    total_cx_gates: int
    total_measurements: int
    total_detectors: int
    
    # Timing
    syndrome_rounds_per_gadget: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "distance": self.star_stats.distance,
            "logical_qubits": self.qft_logical_qubits,
            "clifford_gates": self.clifford_gate_count,
            "non_clifford_gates": self.non_clifford_gate_count,
            "star_gadgets": self.star_gadgets_needed,
            "physical_qubits_per_star": self.star_stats.total_qubits,
            "total_physical_qubits": self.total_physical_qubits,
            "cx_per_star": self.star_stats.cx_gates,
            "total_cx": self.total_cx_gates,
            "measurements_per_star": self.star_stats.measurements,
            "total_measurements": self.total_measurements,
            "detectors_per_star": self.star_stats.detectors,
            "total_detectors": self.total_detectors,
            "ticks_per_gadget": self.syndrome_rounds_per_gadget,
        }


def qft_resource_summary(d: int = 3, stim_dir: Optional[Path] = None) -> QFTResourceSummary:
    """
    Compute complete resource summary for 4-qubit QFT at distance d.
    
    Args:
        d: Code distance (default 3)
        stim_dir: Path to STAR circuit files
        
    Returns:
        QFTResourceSummary with all resource counts
    """
    qft = QFT_4_GATES
    star_stats = parse_star_circuit(d, stim_dir)
    n_gadgets = qft.star_gadgets_needed
    
    return QFTResourceSummary(
        qft_logical_qubits=4,
        clifford_gate_count=len(qft.clifford_gates),
        non_clifford_gate_count=len(qft.non_clifford_gates),
        star_gadgets_needed=n_gadgets,
        star_stats=star_stats,
        total_physical_qubits=n_gadgets * star_stats.total_qubits,
        total_cx_gates=n_gadgets * star_stats.cx_gates,
        total_measurements=n_gadgets * star_stats.measurements,
        total_detectors=n_gadgets * star_stats.detectors,
        syndrome_rounds_per_gadget=star_stats.tick_count,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DIRECT vs INFERRED QUANTITIES
# ══════════════════════════════════════════════════════════════════════════════

DIRECT_QUANTITIES = """
DIRECTLY MEASURED FROM .stim FILES (text parsing)
=================================================
These values are parsed directly from star_d={d}.stim with no estimation:

  Physical Structure:
    • total_qubits        — Number of distinct qubit indices in circuit
    • data_qubits         — d² (geometric, from code distance)
    • ancilla_qubits      — total - data (geometric)

  Gate Counts:
    • h_gates             — Count of H instructions
    • s_gates             — Count of S and S_DAG instructions
    • t_gates             — Count of T and T_DAG instructions
    • cx_gates            — Count of 2-qubit gates (CX/CNOT/CZ/etc)
    • r_resets            — Count of reset operations (R, RX, RY, RZ, R_X, R_Y, R_Z)

  Measurement Structure:
    • measurements        — Count of measurement operations
    • detectors           — Count of DETECTOR annotations
    • observables         — Count of OBSERVABLE_INCLUDE annotations

  Timing:
    • tick_count          — Number of TICK operations (≈ time steps)

  Existing Noise (if present):
    • depolarize1_count   — DEPOLARIZE1 operations
    • depolarize2_count   — DEPOLARIZE2 operations
    • x_error_count       — X_ERROR operations
    • z_error_count       — Z_ERROR operations

  Metadata:
    • total_lines         — Total lines in file
    • instruction_counts  — Frequency of each instruction type
    • unknown_instructions — Instructions not in standard categories
"""

INFERRED_QUANTITIES = """
INFERRED / ESTIMATED (NOT directly measured)
============================================
These would require additional simulation or decoding:

  Error Rates:
    • logical_error_rate  — Needs decoder + many shots
    • threshold_p         — Needs multi-distance fitting
    • pseudothreshold     — Needs fitting at fixed d

  QFT Performance:
    • qft_fidelity        — Depends on all 3 STAR gadgets succeeding
    • output_state_fidelity — Requires full state tomography

  Optimization:
    • optimal_r           — Syndrome rounds, needs threshold analysis
    • break_even_point    — Where QEC helps vs. hurts

  Timing:
    • wall_clock_time     — Hardware-dependent
    • gate_time           — Platform-specific
"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Verification when run directly
# ══════════════════════════════════════════════════════════════════════════════

def print_separator(title: str, char: str = "─", width: int = 70):
    """Print a formatted section separator."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def main(stim_dir: Optional[Path] = None):
    """Run verification of d=3 baseline."""
    
    print("=" * 70)
    print("  STAR Part 4: 4-Qubit QFT — Distance d=3 Baseline")
    print("=" * 70)
    
    # ── Section 1: QFT Gate Structure ─────────────────────────────────────────
    print_separator("1. QFT GATE STRUCTURE")
    
    qft = QFT_4_GATES
    print(f"\nLogical qubits: 4")
    print(f"Total gates: {qft.total_gates}")
    print(f"  Clifford gates: {len(qft.clifford_gates)}")
    print(f"  Non-Clifford gates: {len(qft.non_clifford_gates)} → require STAR")
    
    print("\nClifford gates (no STAR needed):")
    for g in qft.clifford_gates:
        print(f"    {g}")
    
    print("\nNon-Clifford gates (STAR teleportation required):")
    for i, g in enumerate(qft.non_clifford_gates, 1):
        print(f"  [{i}] {g}")
    
    # ── Section 2: STAR Circuit Parsing ───────────────────────────────────────
    print_separator("2. STAR d=3 CIRCUIT STATISTICS")
    
    try:
        stats = parse_star_circuit(3, stim_dir)
        print(f"\nFile: {stats.filepath}")
        print(f"Total lines: {stats.total_lines}")
        
        print(f"\nPhysical structure:")
        print(f"    Total qubits:   {stats.total_qubits}")
        print(f"    Data qubits:    {stats.data_qubits} (d² = 9)")
        print(f"    Ancilla qubits: {stats.ancilla_qubits}")
        
        print(f"\nGate counts:")
        print(f"    H gates:   {stats.h_gates}")
        print(f"    S gates:   {stats.s_gates}")
        print(f"    T gates:   {stats.t_gates}")
        print(f"    CX gates:  {stats.cx_gates}")
        print(f"    Resets:    {stats.r_resets}")
        
        print(f"\nMeasurement structure:")
        print(f"    Measurements: {stats.measurements}")
        print(f"    Detectors:    {stats.detectors}")
        print(f"    Observables:  {stats.observables}")
        
        print(f"\nTiming:")
        print(f"    TICK count: {stats.tick_count}")
        
        if stats.depolarize1_count or stats.depolarize2_count:
            print(f"\nExisting noise operations:")
            print(f"    DEPOLARIZE1: {stats.depolarize1_count}")
            print(f"    DEPOLARIZE2: {stats.depolarize2_count}")
        
        if stats.unknown_instructions:
            print(f"\nNon-standard instructions handled:")
            for instr in sorted(stats.unknown_instructions):
                count = stats.instruction_counts.get(instr, 0)
                print(f"    {instr}: {count}")
        
        print(f"\nInstruction breakdown:")
        for instr, count in sorted(stats.instruction_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"    {instr}: {count}")
        
    except FileNotFoundError as e:
        print(f"\n*** ERROR: {e}")
        print("    Cannot proceed without star_d=3.stim")
        return
    
    # ── Section 3: Total QFT Resources ────────────────────────────────────────
    print_separator("3. TOTAL QFT RESOURCES (d=3)")
    
    summary = qft_resource_summary(3, stim_dir)
    
    print(f"\nSTAR gadgets needed: {summary.star_gadgets_needed}")
    print(f"\nPer-gadget resources:")
    print(f"    Physical qubits: {summary.star_stats.total_qubits}")
    print(f"    CX gates:        {summary.star_stats.cx_gates}")
    print(f"    Measurements:    {summary.star_stats.measurements}")
    print(f"    Detectors:       {summary.star_stats.detectors}")
    
    print(f"\nTotal for 4-qubit QFT:")
    print(f"    Physical qubits: {summary.total_physical_qubits}")
    print(f"    CX gates:        {summary.total_cx_gates}")
    print(f"    Measurements:    {summary.total_measurements}")
    print(f"    Detectors:       {summary.total_detectors}")
    
    # ── Section 4: Direct vs Inferred ─────────────────────────────────────────
    print_separator("4. DIRECT vs INFERRED QUANTITIES")
    print(DIRECT_QUANTITIES)
    print(INFERRED_QUANTITIES)
    
    print("\n" + "=" * 70)
    print("  d=3 baseline verification complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
