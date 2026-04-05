"""
STAR_P1_Emiliano.py


Base level understanding and notes

this code is not just making plots, we are building a small QED system
- QC are fragile, errors can appear with bit flip, phase flip or both 
- we cannot measure whenever bc we will destroy the info we are trying to protect, so we introduce ancillas
-ancillas can tell us that something in the data has changed
- stabalizers are consistency checks that do not tell you the full quantum state but instead weather the structure looks right
- there are x and z stabilizers to cehck weather there are X,Z,Y errors within the Torus


Part 1 circuit for a distance = 3 rotated surface code with exactly two rounds of syndrome
extraction and optional single-qubit error injection between rounds.

- there is zero noise visualization suite

- plot_circuit_diagram()          : horizontal wire circuit diagram (Image 1 style)
- plot_zero_noise_dashboard()     : 6-panel zero-noise summary dashboard
- plot_stabilizer_commutation()   : which Paulis anticommute with which stabilizers
- plot_measurement_outcomes()     : round 1 vs round 2 syndrome bars + detection events
- plot_qubit_state_evolution()    : qubit state at each stage of the circuit
- plot_detection_event_rate()     : per-stabilizer detection rates (zero = healthy)
- plot_cx_flow_diagram()          : CNOT information flow (which ancilla learns what)
- print_zero_noise_report()       : full ASCII zero-noise report to stdout
- run_zero_noise_suite()          : runs everything and saves all plots
- plot_patch()                    : patch geometry
- plot_cx_schedule()              : per-layer CNOT schedule
- plot_syndrome_heatmap()         : shot-by-shot heatmap
- validate_error_signature()      : theory vs simulation table
- build_patch_metadata()          : generalised patch metadata
- compute_detector_reliability()  : cross-seed reliability
- validation_summary_to_json()    : full JSON export
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import argparse
import json

import numpy as np

#plotting
_MPL_ERROR: Exception | None = None
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.lines import Line2D
    _MPL_AVAILABLE = True
except Exception as exc:
    plt = None
    _MPL_AVAILABLE = False
    _MPL_ERROR = exc

#tsim
_IMPORT_ERROR: Exception | None = None
try:
    import tsim  # type: ignore
except Exception as exc:
    tsim = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


# ── type aliases implementation ───────────────────────────────────────────────────────────────
InitState = Literal["plus", "zero"]
Pauli = Literal["X", "Y", "Z"]


@dataclass(frozen=True)
class DataQubitError:
    """A deterministic Pauli error inserted between the two clean rounds."""
    pauli: Pauli
    qubit: int


@dataclass(frozen=True)
class MeasurementFlip:
    """A deterministic ancilla measurement flip in the clean two-round model."""
    ancilla: int
    round_index: int = 2


# d = 3 metadata, physical/code layout declared
DATA_QUBITS: tuple[int, ...] = tuple(range(9))
Z_ANCILLAS: tuple[int, ...] = (9, 10, 11, 12)
X_ANCILLAS: tuple[int, ...] = (13, 14, 15, 16)
ALL_ANCILLAS: tuple[int, ...] = Z_ANCILLAS + X_ANCILLAS
MEASUREMENT_LABELS: tuple[str, ...] = ("Z0", "Z1", "Z2", "Z3", "X0", "X1", "X2", "X3")
# geometry data
QUBIT_COORDS: dict[int, tuple[float, float]] = {
    0: (0.5, 0.5),  1: (1.5, 0.5),  2: (2.5, 0.5),
    3: (0.5, 1.5),  4: (1.5, 1.5),  5: (2.5, 1.5),
    6: (0.5, 2.5),  7: (1.5, 2.5),  8: (2.5, 2.5),
    9:  (1.0, 0.0), 10: (2.0, 1.0), 11: (1.0, 2.0), 12: (2.0, 3.0),
    13: (1.0, 1.0), 14: (3.0, 1.0), 15: (0.0, 2.0), 16: (2.0, 2.0),
}

STABILIZERS: dict[str, dict[str, object]] = {
    "Z0": {"ancilla": 9,  "basis": "Z", "data": [0, 1]},
    "Z1": {"ancilla": 10, "basis": "Z", "data": [1, 2, 4, 5]},
    "Z2": {"ancilla": 11, "basis": "Z", "data": [3, 4, 6, 7]},
    "Z3": {"ancilla": 12, "basis": "Z", "data": [7, 8]},
    "X0": {"ancilla": 13, "basis": "X", "data": [0, 1, 3, 4]},
    "X1": {"ancilla": 14, "basis": "X", "data": [2, 5]},
    "X2": {"ancilla": 15, "basis": "X", "data": [3, 6]},
    "X3": {"ancilla": 16, "basis": "X", "data": [4, 5, 7, 8]},
}

CX_LAYERS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((13, 0),  (14, 2),  (16, 4),  (1, 10), (3, 11), (7, 12)),
    ((13, 3),  (14, 5),  (16, 7),  (2, 10), (4, 11), (8, 12)),
    ((13, 1),  (15, 3),  (16, 5),  (0, 9),  (4, 10), (6, 11)),
    ((13, 4),  (15, 6),  (16, 8),  (1, 9),  (5, 10), (7, 11)),
)

# ── From STAR 
STAR_CIRCUIT_DIR = Path("assets") / "star_circuits"
NOISE_PREFIXES: tuple[str, ...] = (
    "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR",
    "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR",
)


def require_tsim() -> None:
    if tsim is None:
        raise RuntimeError(
            "tsim is not available. Run `uv sync` in the challenge repo."
        ) from _IMPORT_ERROR


def require_matplotlib() -> None:
    if not _MPL_AVAILABLE:
        raise RuntimeError(
            "matplotlib is not available. Run `pip install matplotlib`."
        ) from _MPL_ERROR


def star_circuit_path(distance: int = 3) -> Path:
    return STAR_CIRCUIT_DIR / f"star_d={distance}.stim"


def list_available_star_distances(circuit_dir: Path = STAR_CIRCUIT_DIR) -> list[int]:
    if not circuit_dir.exists():
        return []
    distances: list[int] = []
    for path in circuit_dir.glob("star_d=*.stim"):
        try:
            distances.append(int(path.stem.split("=")[1]))
        except Exception:
            continue
    return sorted(distances)


def load_star_stim_text(distance: int = 3, circuit_dir: Path = STAR_CIRCUIT_DIR) -> str:
    path = circuit_dir / f"star_d={distance}.stim"
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}.")
    return path.read_text(encoding="utf-8")


def strip_noise_from_stim_text(stim_text: str) -> str:
    cleaned: list[str] = []
    for raw_line in stim_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        if stripped.startswith("#"):
            cleaned.append(line)
            continue
        if stripped.startswith("TICK{"):
            cleaned.append("TICK")
            continue
        if any(stripped.startswith(p) for p in NOISE_PREFIXES):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def summarize_reference_stim_text(stim_text: str) -> dict[str, int]:
    summary = {
        "num_lines": 0, "num_ticks": 0, "num_detectors": 0,
        "num_measure_lines": 0, "num_noise_lines": 0,
        "num_feed_forward_lines": 0, "num_rotation_lines": 0,
    }
    for raw_line in stim_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        summary["num_lines"] += 1
        if stripped.startswith("TICK"):          summary["num_ticks"] += 1
        if stripped.startswith("DETECTOR"):     summary["num_detectors"] += 1
        if stripped.startswith(("M ", "MX ")): summary["num_measure_lines"] += 1
        if any(stripped.startswith(p) for p in NOISE_PREFIXES):
            summary["num_noise_lines"] += 1
        if "[FEED_FORWARD]" in stripped:        summary["num_feed_forward_lines"] += 1
        if stripped.startswith("R_Z("):         summary["num_rotation_lines"] += 1
    return summary


# ── clean circuit builder ──────────────────────────────────────────────────────
def _fmt_targets(qubits: tuple[int, ...] | list[int]) -> str:
    return " ".join(str(q) for q in qubits)


def _fmt_pairs(pairs: tuple[tuple[int, int], ...]) -> str:
    flat: list[str] = []
    for a, b in pairs:
        flat.extend([str(a), str(b)])
    return " ".join(flat)


def build_syndrome_round_lines(measurement_flip: MeasurementFlip | None = None) -> list[str]:
    lines = [
        "# Reset ancillas for one syndrome round",
        f"R {_fmt_targets(Z_ANCILLAS)}",
        f"RX {_fmt_targets(X_ANCILLAS)}",
        "TICK",
        "# Entangle ancillas with neighboring data qubits",
    ]
    for layer in CX_LAYERS:
        lines.append(f"CX {_fmt_pairs(layer)}")
        lines.append("TICK")
    if measurement_flip is not None:
        if measurement_flip.ancilla in Z_ANCILLAS:
            lines.append(f"X {measurement_flip.ancilla}")
            lines.append("TICK")
        elif measurement_flip.ancilla in X_ANCILLAS:
            lines.append(f"Z {measurement_flip.ancilla}")
            lines.append("TICK")
        else:
            raise ValueError(f"Ancilla {measurement_flip.ancilla} is not a valid ancilla.")
    lines.extend([
        "# Measure the 8 stabilizers",
        f"M {_fmt_targets(Z_ANCILLAS)}",
        f"MX {_fmt_targets(X_ANCILLAS)}",
        "TICK",
    ])
    return lines

# checking for simple clean basis states
def build_clean_part1_stim_text(
    *,
    init_state: InitState = "plus",
    data_error: DataQubitError | None = None,
    measurement_flip: MeasurementFlip | None = None,
) -> str:
    if init_state not in {"plus", "zero"}:
        raise ValueError("init_state must be 'plus' or 'zero'.")
    lines: list[str] = [
        "# Clean d=3 Part 1 circuit — zero noise baseline",
        "# Two rounds of syndrome extraction",
    ]
    for q in range(17):
        x, y = QUBIT_COORDS[q]
        lines.append(f"QUBIT_COORDS({x}, {y}) {q}")
    if init_state == "plus":
        lines.extend(["# Prepare data in |+>", f"RX {_fmt_targets(DATA_QUBITS)}", "TICK"])
    else:
        lines.extend(["# Prepare data in |0>", f"R {_fmt_targets(DATA_QUBITS)}", "TICK"])

    round1_flip = measurement_flip if (measurement_flip is not None and measurement_flip.round_index == 1) else None
    round2_flip = measurement_flip if (measurement_flip is not None and measurement_flip.round_index == 2) else None

    lines.append("# === ROUND 1 ===")
    lines.extend(build_syndrome_round_lines(measurement_flip=round1_flip))
    if data_error is not None:
        if data_error.qubit not in DATA_QUBITS:
            raise ValueError(f"Data-qubit error must target one of {DATA_QUBITS}.")
        lines.extend([
            f"# Insert deterministic {data_error.pauli} error between rounds",
            f"{data_error.pauli} {data_error.qubit}",
            "TICK",
        ])
    lines.append("# === ROUND 2 ===")
    lines.extend(build_syndrome_round_lines(measurement_flip=round2_flip))
    lines.append("# Detection events = round2 XOR round1")
    for i, label in enumerate(MEASUREMENT_LABELS):
        anc = ALL_ANCILLAS[i]
        x, y = QUBIT_COORDS[anc]
        lines.append(f"DETECTOR({x}, {y}, 1) rec[-{8 - i}] rec[-{16 - i}]  # {label}")
    return "\n".join(lines)


# ── sampling helpers ───────────────────────────────────────────────────────────
def circuit_from_text(stim_text: str):
    require_tsim()
    return tsim.Circuit(stim_text)


def sample_measurements_from_text(stim_text: str, shots: int = 100, seed: int = 42) -> np.ndarray:
    circuit = circuit_from_text(stim_text)
    sampler = circuit.compile_sampler(seed=seed)
    return np.asarray(sampler.sample(shots), dtype=np.int8)


def sample_detectors_from_text(stim_text: str, shots: int = 100, seed: int = 42) -> np.ndarray:
    circuit = circuit_from_text(stim_text)
    sampler = circuit.compile_detector_sampler(seed=seed)
    return np.asarray(sampler.sample(shots), dtype=np.int8)


def sample_clean_measurements(
    *, shots: int = 100, seed: int = 42,
    init_state: InitState = "plus",
    data_error: DataQubitError | None = None,
    measurement_flip: MeasurementFlip | None = None,
) -> np.ndarray:
    stim_text = build_clean_part1_stim_text(
        init_state=init_state, data_error=data_error, measurement_flip=measurement_flip,
    )
    return sample_measurements_from_text(stim_text, shots=shots, seed=seed)


def sample_clean_detectors(
    *, shots: int = 100, seed: int = 42,
    init_state: InitState = "plus",
    data_error: DataQubitError | None = None,
    measurement_flip: MeasurementFlip | None = None,
) -> np.ndarray:
    stim_text = build_clean_part1_stim_text(
        init_state=init_state, data_error=data_error, measurement_flip=measurement_flip,
    )
    return sample_detectors_from_text(stim_text, shots=shots, seed=seed)


def sample_reference_measurements(
    *, distance: int = 3, shots: int = 100, seed: int = 42,
    strip_noise: bool = True, circuit_dir: Path = STAR_CIRCUIT_DIR,
) -> np.ndarray:
    stim_text = load_star_stim_text(distance=distance, circuit_dir=circuit_dir)
    if strip_noise:
        stim_text = strip_noise_from_stim_text(stim_text)
    return sample_measurements_from_text(stim_text, shots=shots, seed=seed)


def sample_reference_detectors(
    *, distance: int = 3, shots: int = 100, seed: int = 42,
    strip_noise: bool = True, circuit_dir: Path = STAR_CIRCUIT_DIR,
) -> np.ndarray:
    stim_text = load_star_stim_text(distance=distance, circuit_dir=circuit_dir)
    if strip_noise:
        stim_text = strip_noise_from_stim_text(stim_text)
    return sample_detectors_from_text(stim_text, shots=shots, seed=seed)


# ── interpretation helpers ─────────────────────────────────────────────────────
def split_round_measurements(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.shape[1] != 16:
        raise ValueError(f"Expected shape (*, 16), got {samples.shape}.")
    return samples[:, :8], samples[:, 8:]


def detection_events_from_measurements(samples: np.ndarray) -> np.ndarray:
    round1, round2 = split_round_measurements(samples)
    return np.bitwise_xor(round1, round2).astype(np.int8)


def parse_clean_measurements(samples: np.ndarray) -> dict[str, np.ndarray]:
    round1, round2 = split_round_measurements(samples)
    det = detection_events_from_measurements(samples)
    return {"round1": round1, "round2": round2, "detection_events": det}


def predict_syndrome_changes(error: DataQubitError) -> list[str]:
    if error.qubit not in DATA_QUBITS:
        raise ValueError(f"Data error must target one of {DATA_QUBITS}, got {error.qubit}")
    flips: list[str] = []
    for label, info in STABILIZERS.items():
        if error.qubit not in info["data"]:
            continue
        if error.pauli in ("X", "Y") and info["basis"] == "Z":
            flips.append(label)
        if error.pauli in ("Z", "Y") and info["basis"] == "X":
            flips.append(label)
    return flips


def format_rates_as_dict(rates: np.ndarray) -> dict[str, float]:
    return {label: float(rate) for label, rate in zip(MEASUREMENT_LABELS, rates.tolist())}



# ZERO-NOISE VISUALIZATION 1 — horizontal wire circuit diagram
#what is the exact time ordered circuit structure

def plot_circuit_diagram(
    save_path: str | Path | None = None,
    show_round_labels: bool = True,
) -> "plt.Figure":
    """
    Horizontal wire circuit diagram matching the Image 1 style.

    Rows = qubits (data top, Z ancillas middle, X ancillas bottom).
    Columns = time steps: init | R1 layers 1-4 | M1 | R2 layers 1-4 | M2.
    CNOT control = filled dot, target = circle with cross.
    Color coding: data wires gray, Z ancilla blue, X ancilla orange.
    """
    require_matplotlib()

    qubit_order = list(range(9)) + list(Z_ANCILLAS) + list(X_ANCILLAS)
    row_of = {q: i for i, q in enumerate(qubit_order)}
    n_qubits = len(qubit_order)

    col_r1  = [2.0, 3.0, 4.0, 5.0]
    col_m1  = 6.5
    col_r2  = [8.0, 9.0, 10.0, 11.0]
    col_m2  = 12.5
    col_end = 13.5

    fig, ax = plt.subplots(figsize=(18, max(7, n_qubits * 0.6)), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(-0.3, col_end + 0.6)
    ax.set_ylim(-0.7, n_qubits + 0.3)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(
        "d=3 Surface Code — Zero-Noise Circuit  (2 syndrome extraction rounds)",
        fontsize=13, fontweight="bold", pad=10,
    )

    wire_colors = {q: "#444" for q in DATA_QUBITS}
    for q in Z_ANCILLAS:
        wire_colors[q] = "#1565C0"
    for q in X_ANCILLAS:
        wire_colors[q] = "#E65100"

    # wires
    for q in qubit_order:
        r = row_of[q]
        ax.hlines(r, 0.0, col_end, colors=wire_colors[q], lw=1.1, alpha=0.55)

    # init labels
    for q in qubit_order:
        r = row_of[q]
        lbl = "|0⟩" if q in Z_ANCILLAS else "|+⟩"
        ax.text(-0.08, r, lbl, ha="right", va="center", fontsize=8,
                color=wire_colors[q], fontweight="bold")
        ax.text(0.02, r + 0.32, str(q), ha="left", va="top", fontsize=5.5, color="#aaa")

    def draw_cnot(col: float, ctrl_q: int, tgt_q: int) -> None:
        cr, tr = row_of[ctrl_q], row_of[tgt_q]
        c = wire_colors[ctrl_q]
        ax.vlines(col, min(cr, tr), max(cr, tr), colors=c, lw=1.1)
        ax.plot(col, cr, "o", ms=7, color=c, zorder=5)
        circ = plt.Circle((col, tr), 0.17, color="white", ec=c, lw=1.4, zorder=4)
        ax.add_patch(circ)
        ax.plot([col - 0.17, col + 0.17], [tr, tr], color=c, lw=1.1, zorder=5)
        ax.plot([col, col], [tr - 0.17, tr + 0.17], color=c, lw=1.1, zorder=5)

    def draw_measure(col: float, q: int, label: str = "") -> None:
        r = row_of[q]
        c = wire_colors[q]
        box = FancyBboxPatch(
            (col - 0.27, r - 0.27), 0.54, 0.54,
            boxstyle="round,pad=0.04",
            facecolor="white", edgecolor=c, lw=1.2, zorder=6,
        )
        ax.add_patch(box)
        ax.annotate(
            "", xy=(col + 0.15, r - 0.1), xytext=(col - 0.17, r + 0.1),
            arrowprops=dict(arrowstyle="-|>", color=c, lw=0.9, mutation_scale=7),
            zorder=7,
        )
        if label:
            ax.text(col, r + 0.43, label, ha="center", va="bottom",
                    fontsize=5.5, color=c)

    for layer_i, layer_pairs in enumerate(CX_LAYERS):
        for ctrl, tgt in layer_pairs:
            draw_cnot(col_r1[layer_i], ctrl, tgt)

    for q in Z_ANCILLAS:
        draw_measure(col_m1, q)
    for q in X_ANCILLAS:
        draw_measure(col_m1, q)

    for layer_i, layer_pairs in enumerate(CX_LAYERS):
        for ctrl, tgt in layer_pairs:
            draw_cnot(col_r2[layer_i], ctrl, tgt)

    for i, q in enumerate(Z_ANCILLAS):
        draw_measure(col_m2, q, MEASUREMENT_LABELS[i])
    for i, q in enumerate(X_ANCILLAS):
        draw_measure(col_m2, q, MEASUREMENT_LABELS[4 + i])

    if show_round_labels:
        mid = (col_m1 + col_r2[0]) / 2
        ax.axvline(mid, color="#bbb", lw=1, ls="--", alpha=0.5)
        ax.text(sum(col_r1) / 4, -0.55, "Round 1", ha="center",
                fontsize=9, color="#555", fontstyle="italic")
        ax.text(sum(col_r2) / 4, -0.55, "Round 2", ha="center",
                fontsize=9, color="#555", fontstyle="italic")
        ax.text(col_m1, -0.55, "M₁", ha="center", fontsize=8, color="#888")
        ax.text(col_m2, -0.55, "M₂", ha="center", fontsize=8, color="#888")

    for i, col in enumerate(col_r1 + col_r2):
        ax.text(col, n_qubits + 0.15, f"L{(i % 4) + 1}", ha="center",
                fontsize=7, color="#bbb")

    legend_elements = [
        Line2D([0], [0], color="#444",    lw=2, label="Data qubit wire"),
        Line2D([0], [0], color="#1565C0", lw=2, label="Z ancilla (measures Z stab.)"),
        Line2D([0], [0], color="#E65100", lw=2, label="X ancilla (measures X stab.)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#444", ms=8,
               label="CNOT control"),
        mpatches.Patch(facecolor="white", edgecolor="#444", label="CNOT target ⊕"),
        mpatches.Patch(facecolor="white", edgecolor="#555", label="Measurement"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8,
              framealpha=0.95, edgecolor="#ccc", ncol=2)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Circuit diagram saved to {save_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ZERO-NOISE VISUALIZATION 2 — detection event rate bar chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_detection_event_rate(
    samples: np.ndarray | None = None,
    shots: int = 1000,
    seed: int = 42,
    data_error: DataQubitError | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    ax: "plt.Axes | None" = None,
) -> "plt.Figure | None":
    """
    Bar chart of detection event rates per stabilizer.

    Zero noise: all bars at 0.000 — this is your baseline.
    Predicted flips (from theory) are marked with a red border.
    """
    require_matplotlib()

    if samples is None:
        samples = sample_clean_measurements(shots=shots, seed=seed, data_error=data_error)

    det = detection_events_from_measurements(samples)
    rates = np.mean(det, axis=0)
    predicted = set(predict_syndrome_changes(data_error)) if data_error else set()

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
    else:
        fig = None

    ax.set_facecolor("white")
    colors      = ["#1976D2" if l.startswith("Z") else "#F57C00" for l in MEASUREMENT_LABELS]
    edge_colors = ["#C62828" if l in predicted else c for l, c in zip(MEASUREMENT_LABELS, colors)]
    lwidths     = [3 if l in predicted else 0.5 for l in MEASUREMENT_LABELS]

    bars = ax.bar(MEASUREMENT_LABELS, rates, color=colors, edgecolor=edge_colors,
                  linewidth=lwidths, alpha=0.85, zorder=3)

    ax.axhline(0, color="#333", lw=0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Detection event rate", fontsize=10)
    ax.set_xlabel("Stabilizer", fontsize=10)
    ax.yaxis.grid(True, alpha=0.4, lw=0.5)
    ax.set_axisbelow(True)

    for bar, rate in zip(bars, rates):
        if rate > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                    f"{rate:.3f}", ha="center", va="bottom", fontsize=8)

    ax.text(0.01, 0.96, "Zero noise  →  all rates = 0.000",
            transform=ax.transAxes, fontsize=8, color="#2E7D32",
            va="top", style="italic")

    t = title or ("Zero-noise detection rates" if data_error is None
                  else f"Detection rates — {data_error.pauli} error on q{data_error.qubit}")
    ax.set_title(t, fontsize=11, fontweight="bold")

    legend_elements = [
        mpatches.Patch(color="#1976D2", alpha=0.85, label="Z stabilizer"),
        mpatches.Patch(color="#F57C00", alpha=0.85, label="X stabilizer"),
    ]
    if predicted:
        legend_elements.append(
            mpatches.Patch(facecolor="#E53935", alpha=0.3,
                           edgecolor="#C62828", lw=2, label="Predicted flip (theory)")
        )
    ax.legend(handles=legend_elements, fontsize=8, framealpha=0.9)

    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"Detection rate plot saved to {save_path}")
        return fig
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ZERO-NOISE VISUALIZATION 3 — round 1 vs round 2 grouped bar chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_measurement_outcomes(
    samples: np.ndarray | None = None,
    shots: int = 1000,
    seed: int = 42,
    data_error: DataQubitError | None = None,
    save_path: str | Path | None = None,
    ax: "plt.Axes | None" = None,
) -> "plt.Figure | None":
    """
    Grouped bar chart: Round 1 vs Round 2 syndrome + detection events.

    Zero noise: R1 and R2 bars are identical → red detection bars = 0.
    With an error between rounds: bars diverge and detection bars rise to 1.
    """
    require_matplotlib()

    if samples is None:
        samples = sample_clean_measurements(shots=shots, seed=seed, data_error=data_error)

    r1, r2 = split_round_measurements(samples)
    r1_rates  = np.mean(r1, axis=0)
    r2_rates  = np.mean(r2, axis=0)
    det_rates = np.mean(detection_events_from_measurements(samples), axis=0)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(11, 4.5), facecolor="white")
    else:
        fig = None

    ax.set_facecolor("white")
    x, w = np.arange(8), 0.27

    ax.bar(x - w, r1_rates,  width=w, label="Round 1",
           color="#42A5F5", alpha=0.85, zorder=3)
    ax.bar(x,     r2_rates,  width=w, label="Round 2",
           color="#1565C0", alpha=0.85, zorder=3)
    ax.bar(x + w, det_rates, width=w, label="Detection (R2 XOR R1)",
           color="#E53935", alpha=0.85, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(MEASUREMENT_LABELS, fontsize=9)
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("Rate (fraction of shots)", fontsize=10)
    ax.set_xlabel("Stabilizer", fontsize=10)
    ax.yaxis.grid(True, alpha=0.4, lw=0.5)
    ax.set_axisbelow(True)

    title = "Round 1 vs Round 2 syndrome — " + (
        "zero noise  (R1 = R2, detection = 0)" if data_error is None
        else f"{data_error.pauli} error on q{data_error.qubit}"
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.text(0.01, 0.97, "Zero noise: Round 1 \u2261 Round 2  \u2192  red bars all zero",
            transform=ax.transAxes, fontsize=8, color="#2E7D32",
            va="top", style="italic")

    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"Measurement outcomes plot saved to {save_path}")
        return fig
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ZERO-NOISE VISUALIZATION 4 — stabilizer commutation table
# ══════════════════════════════════════════════════════════════════════════════
def plot_stabilizer_commutation(
    save_path: str | Path | None = None,
    ax: "plt.Axes | None" = None,
) -> "plt.Figure | None":
    """
    Heatmap: which single-qubit Pauli errors anticommute with which stabilizers.

    Green cell = error commutes with stabilizer (no detection).
    Red cell   = error anticommutes (stabilizer flips, detection fires).

    This is your complete theoretical ground truth for error diagnosis.
    """
    require_matplotlib()

    error_labels = [f"{p}\u00b7q{q}" for q in DATA_QUBITS for p in ["X", "Y", "Z"]]
    n_errors = len(error_labels)

    matrix = np.zeros((n_errors, 8), dtype=int)
    for ei, (q, p) in enumerate([(q, p) for q in DATA_QUBITS for p in ["X", "Y", "Z"]]):
        for fi, label in enumerate(MEASUREMENT_LABELS):
            if label in predict_syndrome_changes(DataQubitError(p, q)):  # type: ignore[arg-type]
                matrix[ei, fi] = 1

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 9), facecolor="white")
    else:
        fig = None

    ax.set_facecolor("white")
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(8))
    ax.set_xticklabels(MEASUREMENT_LABELS, fontsize=10, fontweight="bold")
    ax.set_yticks(range(n_errors))
    ax.set_yticklabels(error_labels, fontsize=8)
    ax.set_xlabel("Stabilizer", fontsize=11)
    ax.set_ylabel("Error (Pauli \u00b7 data qubit)", fontsize=11)
    ax.set_title(
        "Stabilizer commutation table\n"
        "Red = anticommutes (detection fires)     Green = commutes (silent)",
        fontsize=11, fontweight="bold",
    )

    for ei in range(n_errors):
        for fi in range(8):
            v = matrix[ei, fi]
            ax.text(fi, ei, "\u2717" if v else "\u00b7",
                    ha="center", va="center", fontsize=9,
                    color="white" if v else "#aaa",
                    fontweight="bold" if v else "normal")

    for q_i in range(1, 9):
        ax.axhline(q_i * 3 - 0.5, color="white", lw=1.5)
    ax.axvline(3.5, color="white", lw=2)
    ax.text(1.5, -1.0, "Z stabilizers", ha="center", fontsize=9,
            color="#1565C0", fontweight="bold")
    ax.text(5.5, -1.0, "X stabilizers", ha="center", fontsize=9,
            color="#E65100", fontweight="bold")

    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"Commutation table saved to {save_path}")
        return fig
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ZERO-NOISE VISUALIZATION 5 — qubit state evolution table
# ══════════════════════════════════════════════════════════════════════════════
def plot_qubit_state_evolution(
    save_path: str | Path | None = None,
) -> "plt.Figure":
    """
    Grid table showing the conceptual state of each qubit class at each
    circuit stage from initialization through both syndrome rounds.

    Zero noise: ancilla states are deterministic after measurement.
    Detection events column: all zeros in the clean case.
    """
    require_matplotlib()

    stages = [
        "Init", "Reset\nancillas",
        "CX\nlayer 1", "CX\nlayer 2", "CX\nlayer 3", "CX\nlayer 4",
        "Measure\nM\u2081",
        "Reset\nancillas",
        "CX layers\n1\u20134 (R2)",
        "Measure\nM\u2082",
    ]

    rows = [
        {
            "label": "Data\nq0\u2013q8",
            "color": "#444",
            "states": [
                "|+\u27e9", "|+\u27e9", "|+\u27e9\n(entangled)", "|+\u27e9\n(entangled)",
                "|+\u27e9\n(entangled)", "|+\u27e9\n(entangled)", "|+\u27e9\nunchanged",
                "|+\u27e9", "|+\u27e9\n(entangled)", "|+\u27e9\nunchanged",
            ],
        },
        {
            "label": "Z ancillas\nq9\u2013q12",
            "color": "#1565C0",
            "states": [
                "\u2014", "|0\u27e9", "|0\u27e9", "accum.\nZ parity",
                "accum.\nZ parity", "parity\ncomplete",
                "m\u2208{0,1}\n(syndrome)", "|0\u27e9 reset", "accum.\nZ parity",
                "m\u2208{0,1}\n(syndrome)",
            ],
        },
        {
            "label": "X ancillas\nq13\u2013q16",
            "color": "#E65100",
            "states": [
                "\u2014", "|+\u27e9", "accum.\nX parity", "accum.\nX parity",
                "accum.\nX parity", "parity\ncomplete",
                "m\u2208{0,1}\n(syndrome)", "|+\u27e9 reset", "accum.\nX parity",
                "m\u2208{0,1}\n(syndrome)",
            ],
        },
        {
            "label": "Detection\nevents",
            "color": "#555",
            "states": [
                "\u2014", "\u2014", "\u2014", "\u2014", "\u2014", "\u2014",
                "\u2014", "\u2014", "\u2014",
                "M\u2082 XOR M\u2081\n\u2192 0 (clean)\nor 1 (error!)",
            ],
        },
    ]

    n_rows, n_cols = len(rows), len(stages)
    fig, ax = plt.subplots(figsize=(max(18, n_cols * 1.9), n_rows * 2.2), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(
        "Qubit state evolution — zero-noise circuit\n"
        "(conceptual: actual states are quantum superpositions)",
        fontsize=12, fontweight="bold", pad=8,
    )

    for ci, stage in enumerate(stages):
        ax.text(ci + 0.5, -0.18, stage, ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#333", multialignment="center")
        if ci % 2 == 0:
            ax.add_patch(mpatches.FancyBboxPatch(
                (ci, 0), 1.0, n_rows,
                boxstyle="square,pad=0",
                facecolor="#f8f8f8", edgecolor="none", zorder=0,
            ))

    for ri, row in enumerate(rows):
        ax.text(-0.06, ri + 0.5, row["label"], ha="right", va="center",
                fontsize=9, fontweight="bold", color=row["color"],
                multialignment="right")
        for ci, txt in enumerate(row["states"]):
            is_measure = ci in (6, 9)
            is_detect  = ri == 3 and ci == 9
            bg = "#FFF9C4" if is_measure else ("#FFEBEE" if is_detect else "white")
            ax.add_patch(mpatches.FancyBboxPatch(
                (ci + 0.03, ri + 0.05), 0.94, 0.90,
                boxstyle="round,pad=0.03",
                facecolor=bg, edgecolor="#ddd", lw=0.5, zorder=1,
            ))
            ax.text(ci + 0.5, ri + 0.5, txt,
                    ha="center", va="center", fontsize=7.5,
                    color=row["color"] if txt != "\u2014" else "#bbb",
                    multialignment="center", zorder=2)

    ax.axvline(7, color="#aaa", lw=1.5, ls="--", alpha=0.6)
    ax.text(3.5, n_rows + 0.12, "Round 1", ha="center",
            fontsize=10, color="#555", fontstyle="italic")
    ax.text(8.5, n_rows + 0.12, "Round 2", ha="center",
            fontsize=10, color="#555", fontstyle="italic")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"State evolution table saved to {save_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ZERO-NOISE VISUALIZATION 6 — CX information flow
# ══════════════════════════════════════════════════════════════════════════════
def plot_cx_flow_diagram(
    save_path: str | Path | None = None,
) -> "plt.Figure":
    """
    Four-panel diagram: which data qubits feed which ancilla each CX layer.

    Left column = data (information source).
    Right column = ancilla (accumulates parity).
    Arrow thickness and color indicate the layer.
    Dimmed qubits are inactive in that layer.
    """
    require_matplotlib()

    layer_colors = ["#1E88E5", "#43A047", "#FB8C00", "#E53935"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 6.5), facecolor="white")
    fig.suptitle(
        "CNOT information flow per layer — what each ancilla learns from which data qubits",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for layer_i, (ax, color) in enumerate(zip(axes, layer_colors)):
        ax.set_facecolor("white")
        ax.set_xlim(-0.3, 2.4)
        ax.set_ylim(-0.5, 9.5)
        ax.axis("off")
        ax.set_title(f"Layer {layer_i + 1}", fontsize=11, fontweight="bold", color=color)

        data_y    = {q: float(8 - q) for q in DATA_QUBITS}
        all_anc_y = {}
        for i, q in enumerate(Z_ANCILLAS + X_ANCILLAS):
            all_anc_y[q] = 8.2 - i * 1.05

        active: set[int] = set()
        for ctrl, tgt in CX_LAYERS[layer_i]:
            active.add(ctrl)
            active.add(tgt)

        # data qubits
        for q in DATA_QUBITS:
            y  = data_y[q]
            c  = "#222" if q in active else "#ccc"
            ax.plot(0, y, "o", ms=14, color=c, zorder=4,
                    markeredgecolor="white", markeredgewidth=1)
            ax.text(-0.1, y, f"d{q}", ha="right", va="center",
                    fontsize=8, color=c, fontweight="bold")

        # ancilla qubits
        for i, q in enumerate(Z_ANCILLAS + X_ANCILLAS):
            y    = all_anc_y[q]
            is_z = q in Z_ANCILLAS
            base = "#1976D2" if is_z else "#F57C00"
            c    = base if q in active else "#ddd"
            ax.plot(2.0, y, "D", ms=12, color=c, zorder=4,
                    markeredgecolor="white", markeredgewidth=1)
            name = f"Z{Z_ANCILLAS.index(q)}" if is_z else f"X{X_ANCILLAS.index(q)}"
            ax.text(2.1, y, name, ha="left", va="center",
                    fontsize=7.5, color=c, fontweight="bold")

        # arrows
        for ctrl, tgt in CX_LAYERS[layer_i]:
            if ctrl in X_ANCILLAS:
                x0, y0 = 0.14, data_y[tgt]
                x1, y1 = 1.86, all_anc_y[ctrl]
            else:
                x0, y0 = 0.14, data_y[ctrl]
                x1, y1 = 1.86, all_anc_y[tgt]
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>", color=color,
                    lw=1.8, mutation_scale=11,
                    shrinkA=7, shrinkB=7,
                    connectionstyle="arc3,rad=0.08",
                ),
                zorder=3,
            )

        ax.text(0.0,  9.4, "Data",    ha="center", fontsize=9, color="#444", fontweight="bold")
        ax.text(2.0,  9.4, "Ancilla", ha="center", fontsize=9, color="#555", fontweight="bold")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"CX flow diagram saved to {save_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ZERO-NOISE VISUALIZATION 7 — syndrome heatmap (enhanced)
# ══════════════════════════════════════════════════════════════════════════════
def plot_syndrome_heatmap(
    samples: np.ndarray,
    title: str = "Syndrome heatmap",
    save_path: str | Path | None = None,
    max_shots_shown: int = 200,
    ax: "plt.Axes | None" = None,
) -> "plt.Figure | None":
    """
    Heatmap of detection events: stabilizers on y-axis, shots on x-axis.

    Zero noise: entirely green — no detection events fire anywhere.
    Any red cell indicates a detection event in that shot for that stabilizer.
    """
    require_matplotlib()

    det = detection_events_from_measurements(samples)
    n_shots  = min(det.shape[0], max_shots_shown)
    det_view = det[:n_shots, :]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(min(14, n_shots / 8 + 2), 3.5), facecolor="white")
    else:
        fig = None

    ax.set_facecolor("white")
    im = ax.imshow(
        det_view.T, aspect="auto", interpolation="nearest",
        cmap="RdYlGn_r", vmin=0, vmax=1,
        extent=[-0.5, n_shots - 0.5, 7.5, -0.5],
    )
    ax.set_yticks(range(8))
    ax.set_yticklabels(MEASUREMENT_LABELS, fontsize=9)
    ax.set_xlabel("Shot index", fontsize=10)
    ax.set_ylabel("Stabilizer", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    avg = det_view.mean(axis=0)
    for i, rate in enumerate(avg):
        ax.text(n_shots + 0.8, i, f"{rate:.3f}", va="center", fontsize=8, color="#333")
    ax.text(n_shots + 0.8, -0.8, "avg", fontsize=8, color="#666", style="italic")

    if own_fig:
        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.08, label="Detection event")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"Syndrome heatmap saved to {save_path}")
        return fig
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ZERO-NOISE VISUALIZATION 8 — six-panel dashboard
# ══════════════════════════════════════════════════════════════════════════════
def plot_zero_noise_dashboard(
    shots: int = 500,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> "plt.Figure":
    """
    Six-panel zero-noise vs error comparison dashboard.

    Left column: zero-noise baseline.
    Right column: X error on q4 (simplest non-trivial case).

    [A] Detection rates zero    [B] Detection rates X·q4
    [C] R1 vs R2 zero           [D] R1 vs R2 X·q4
    [E] Heatmap zero            [F] Heatmap X·q4
    """
    require_matplotlib()

    samples_clean = sample_clean_measurements(shots=shots, seed=seed)
    samples_x4    = sample_clean_measurements(
        shots=shots, seed=seed, data_error=DataQubitError("X", 4)
    )

    fig = plt.figure(figsize=(18, 13), facecolor="white")
    fig.suptitle(
        "Zero-noise baseline dashboard — d=3 rotated surface code\n"
        "Left: zero noise   |   Right: X error on q4 (comparison)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    plot_detection_event_rate(samples=samples_clean,
                               title="[A] Detection rates — ZERO NOISE", ax=ax_a)
    plot_detection_event_rate(samples=samples_x4, data_error=DataQubitError("X", 4),
                               title="[B] Detection rates — X error on q4", ax=ax_b)
    plot_measurement_outcomes(samples=samples_clean, ax=ax_c)
    plot_measurement_outcomes(samples=samples_x4, data_error=DataQubitError("X", 4), ax=ax_d)
    plot_syndrome_heatmap(samples_clean,
                          title="[E] Heatmap — ZERO NOISE (all green = healthy)",
                          ax=ax_e, max_shots_shown=200)
    plot_syndrome_heatmap(samples_x4,
                          title="[F] Heatmap — X error on q4",
                          ax=ax_f, max_shots_shown=200)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Dashboard saved to {save_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ZERO-NOISE ASCII REPORT
# ══════════════════════════════════════════════════════════════════════════════
def print_zero_noise_report(shots: int = 1000, seed: int = 42) -> None:
    """
    Full ASCII zero-noise status report printed to stdout.

    Covers circuit structure, stabilizer definitions, CX schedule,
    sampled syndrome values, detection event rates, all 27 single-qubit
    error predictions, and a pass/fail summary.
    """
    sep = "=" * 72

    print(f"\n{sep}")
    print("  TEAM STAR — PART 1 — ZERO-NOISE CIRCUIT REPORT")
    print(f"  d=3 rotated surface code | {shots} shots | seed={seed}")
    print(sep)

    print("\n\u2500\u2500 CIRCUIT STRUCTURE \u2500" + "\u2500" * 52)
    print(f"  Distance            : 3")
    print(f"  Data qubits         : {len(DATA_QUBITS)}  (q0-q8, initialized |+\u27e9)")
    print(f"  Z ancillas          : {len(Z_ANCILLAS)}  (q9-q12, |0\u27e9, measure Z stabilizers)")
    print(f"  X ancillas          : {len(X_ANCILLAS)}  (q13-q16, |+\u27e9, measure X stabilizers)")
    print(f"  Total qubits        : {len(DATA_QUBITS) + len(ALL_ANCILLAS)}")
    print(f"  Syndrome rounds     : 2  (clean, zero noise)")
    print(f"  CX layers / round   : 4")
    print(f"  Total measurements  : 16  (8 per round)")
    print(f"  Detectors defined   : 8  (round2 XOR round1 per stabilizer)")
    print(f"  Noise model         : NONE \u2014 clean baseline")

    print("\n\u2500\u2500 STABILIZER DEFINITIONS \u2500" + "\u2500" * 47)
    print(f"  {'Label':<6} {'Basis':<7} {'Ancilla':<9} {'Data qubits':<22} Detects")
    print("  " + "\u2500" * 58)
    for label in MEASUREMENT_LABELS:
        info = STABILIZERS[label]
        detects = "X-type bit-flip errors" if info["basis"] == "Z" else "Z-type phase-flip errors"
        print(f"  {label:<6} {info['basis']:<7} q{info['ancilla']:<8} "
              f"{str(info['data']):<22} {detects}")

    print("\n\u2500\u2500 CX SCHEDULE \u2500" + "\u2500" * 57)
    for i, layer in enumerate(CX_LAYERS):
        x_pairs = [(c, t) for c, t in layer if c in X_ANCILLAS]
        z_pairs = [(c, t) for c, t in layer if t in Z_ANCILLAS]
        print(f"  Layer {i+1}:")
        if x_pairs:
            print("    X-checks (ancilla\u2192data): "
                  + ", ".join(f"q{c}\u2192q{t}" for c, t in x_pairs))
        if z_pairs:
            print("    Z-checks (data\u2192ancilla): "
                  + ", ".join(f"q{c}\u2192q{t}" for c, t in z_pairs))

    print("\n\u2500\u2500 ZERO-NOISE SAMPLING RESULTS \u2500" + "\u2500" * 42)
    samples  = sample_clean_measurements(shots=shots, seed=seed)
    r1, r2   = split_round_measurements(samples)
    det      = detection_events_from_measurements(samples)
    r1_rates = np.mean(r1, axis=0)
    r2_rates = np.mean(r2, axis=0)
    det_rates= np.mean(det, axis=0)

    print(f"\n  {'Stabilizer':<12} {'R1 rate':<12} {'R2 rate':<12} {'Det rate':<12} Status")
    print("  " + "\u2500" * 56)
    all_zero = True
    for i, label in enumerate(MEASUREMENT_LABELS):
        ok = det_rates[i] < 0.001
        if not ok:
            all_zero = False
        print(f"  {label:<12} {r1_rates[i]:<12.4f} {r2_rates[i]:<12.4f} "
              f"{det_rates[i]:<12.4f} {'OK' if ok else 'NON-ZERO'}")

    print()
    print("  " + ("ALL DETECTION RATES = 0.000 \u2014 zero-noise baseline confirmed."
                  if all_zero else
                  "WARNING: Non-zero detection rates \u2014 check circuit construction."))

    print("\n\u2500\u2500 SINGLE-SHOT EXAMPLE (shot 0) \u2500" + "\u2500" * 40)
    print(f"  {'Stabilizer':<12} {'Round 1':<10} {'Round 2':<10} Detection")
    print("  " + "\u2500" * 42)
    for i, label in enumerate(MEASUREMENT_LABELS):
        dv = int(det[0, i])
        flag = "  \u2190 EVENT" if dv else ""
        print(f"  {label:<12} {int(r1[0,i]):<10} {int(r2[0,i]):<10} {dv}{flag}")

    print("\n\u2500\u2500 ERROR SIGNATURE PREDICTIONS (theory) \u2500" + "\u2500" * 33)
    print(f"  {'Error':<12} Flipped stabilizers")
    print("  " + "\u2500" * 48)
    for q in DATA_QUBITS:
        for p in ["X", "Y", "Z"]:
            flips = predict_syndrome_changes(DataQubitError(p, q))  # type: ignore[arg-type]
            print(f"  {p}\u00b7q{q:<8}  {', '.join(flips) if flips else '(none)'}")

    print(f"\n{sep}")
    print("  RESULT: " + ("PASS \u2014 zero-noise circuit is clean.\n"
                           "  Next step: introduce noise and compare against this baseline."
                           if all_zero else
                           "FAIL \u2014 unexpected events in zero-noise run."))
    print(sep + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_zero_noise_suite(
    shots: int = 500,
    seed: int = 42,
    output_dir: str | Path = "zero_noise_plots",
) -> dict[str, Path]:
    """
    Run the complete zero-noise visualization suite.

    Prints the ASCII report and saves 11 plots to output_dir.
    Returns a dict mapping name -> path for every file saved.
    """
    require_matplotlib()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    print(f"\nGenerating zero-noise suite \u2192 {out}/")
    print_zero_noise_report(shots=shots, seed=seed)

    plots = [
        ("01_circuit_diagram.png",     lambda p: plot_circuit_diagram(save_path=p)),
        ("02_patch_geometry.png",      lambda p: plot_patch(save_path=p)),
        ("03_patch_X_error_q4.png",    lambda p: plot_patch(
            save_path=p, highlight_error=DataQubitError("X", 4))),
        ("04_cx_schedule.png",         lambda p: plot_cx_schedule(save_path=p)),
        ("05_cx_flow.png",             lambda p: plot_cx_flow_diagram(save_path=p)),
        ("06_state_evolution.png",     lambda p: plot_qubit_state_evolution(save_path=p)),
        ("07_commutation_table.png",   lambda p: plot_stabilizer_commutation(save_path=p)),
        ("08_detection_rates_zero.png",lambda p: plot_detection_event_rate(
            shots=shots, seed=seed,
            title="Detection event rates \u2014 ZERO NOISE baseline", save_path=p)),
        ("09_measurement_outcomes.png",lambda p: plot_measurement_outcomes(
            shots=shots, seed=seed, save_path=p)),
        ("10_heatmap_zero.png",        lambda p: plot_syndrome_heatmap(
            sample_clean_measurements(shots=shots, seed=seed),
            title="Syndrome heatmap \u2014 ZERO NOISE (all green = healthy)",
            save_path=p)),
        ("11_dashboard.png",           lambda p: plot_zero_noise_dashboard(
            shots=shots, seed=seed, save_path=p)),
    ]

    for fname, fn in plots:
        p = out / fname
        fn(p)
        saved[fname.split(".")[0][3:]] = p

    print(f"\n\u2713 Zero-noise suite complete. {len(saved)} plots saved to {out}/")
    for name, path in saved.items():
        print(f"  {name:<35} \u2192 {path.name}")
    return saved


# ══════════════════════════════════════════════════════════════════════════════
# PREVIOUSLY ADDED ENHANCEMENTS (retained, compact)
# ══════════════════════════════════════════════════════════════════════════════
def plot_patch(
    save_path: str | Path | None = None,
    highlight_error: DataQubitError | None = None,
) -> "plt.Figure":
    require_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
    ax.set_facecolor("white"); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("d=3 Rotated Surface Code Patch", fontsize=14, fontweight="bold", pad=12)
    for label, info in STABILIZERS.items():
        color = "#1976D2" if info["basis"] == "Z" else "#F57C00"
        verts = [QUBIT_COORDS[info["ancilla"]]] + [QUBIT_COORDS[d] for d in info["data"]]
        ax.add_patch(plt.Polygon(verts, closed=True, color=color, alpha=0.25, zorder=1))
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=8,
                color=color, fontweight="bold", zorder=5)
    drawn: set[frozenset] = set()
    for info in STABILIZERS.values():
        for d in info["data"]:
            key = frozenset({info["ancilla"], d})
            if key not in drawn:
                drawn.add(key)
                x0, y0 = QUBIT_COORDS[info["ancilla"]]
                x1, y1 = QUBIT_COORDS[d]
                ax.plot([x0, x1], [y0, y1], "-", color="#999", lw=0.8, zorder=2)
    grid = {(c, r): c + r * 3 for r in range(3) for c in range(3)}
    for (c, r), qid in grid.items():
        for dc, dr in [(1, 0), (0, 1)]:
            nb = grid.get((c + dc, r + dr))
            if nb is not None:
                x0, y0 = QUBIT_COORDS[qid]; x1, y1 = QUBIT_COORDS[nb]
                ax.plot([x0, x1], [y0, y1], "-", color="#bbb", lw=1.2, zorder=2)
    for anc in Z_ANCILLAS:
        x, y = QUBIT_COORDS[anc]
        ax.plot(x, y, "D", ms=14, color="#1976D2", zorder=6,
                markeredgecolor="white", markeredgewidth=1)
        ax.text(x, y, f"q{anc}", ha="center", va="center",
                fontsize=6, color="white", zorder=7, fontweight="bold")
    for anc in X_ANCILLAS:
        x, y = QUBIT_COORDS[anc]
        ax.plot(x, y, "D", ms=14, color="#F57C00", zorder=6,
                markeredgecolor="white", markeredgewidth=1)
        ax.text(x, y, f"q{anc}", ha="center", va="center",
                fontsize=6, color="white", zorder=7, fontweight="bold")
    pauli_colors = {"X": "#D32F2F", "Y": "#7B1FA2", "Z": "#388E3C"}
    errored = highlight_error.qubit if highlight_error else None
    for qid in DATA_QUBITS:
        x, y = QUBIT_COORDS[qid]
        c  = pauli_colors.get(highlight_error.pauli, "#222") if qid == errored else "#222"
        rg = "#FFD600" if qid == errored else "white"
        ax.plot(x, y, "o", ms=22, color=c, zorder=8, markeredgecolor=rg, markeredgewidth=2)
        ax.text(x, y, f"q{qid}", ha="center", va="center",
                fontsize=7, color="white", zorder=9, fontweight="bold")
    if highlight_error:
        x, y = QUBIT_COORDS[highlight_error.qubit]
        ax.text(x, y + 0.3, f"{highlight_error.pauli} error",
                ha="center", va="bottom", fontsize=8,
                color=pauli_colors[highlight_error.pauli], fontweight="bold", zorder=10)
    ax.legend(handles=[
        mpatches.Patch(facecolor="#222",    label="Data qubit"),
        mpatches.Patch(facecolor="#1976D2", label="Z ancilla"),
        mpatches.Patch(facecolor="#F57C00", label="X ancilla"),
        mpatches.Patch(facecolor="#1976D2", alpha=0.25, label="Z stabilizer face"),
        mpatches.Patch(facecolor="#F57C00", alpha=0.25, label="X stabilizer face"),
    ], loc="upper right", fontsize=8, framealpha=0.9, edgecolor="#ccc")
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, 3.5); ax.invert_yaxis()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Patch diagram saved to {save_path}")
    return fig


def plot_cx_schedule(
    save_path: str | Path | None = None,
    layer_indices: list[int] | None = None,
) -> "plt.Figure":
    require_matplotlib()
    layers = layer_indices or list(range(4))
    fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 4.5), facecolor="white")
    if len(layers) == 1:
        axes = [axes]
    for layer_i, ax in zip(layers, axes):
        ax.set_facecolor("white"); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"Layer {layer_i + 1}", fontsize=11, fontweight="bold")
        active = {q for pair in CX_LAYERS[layer_i] for q in pair}
        for qid in range(17):
            x, y = QUBIT_COORDS[qid]
            dim = qid not in active
            if qid in DATA_QUBITS:
                ax.plot(x, y, "o", ms=16, color="#bbb" if dim else "#222", zorder=4,
                        markeredgecolor="white", markeredgewidth=1)
                ax.text(x, y, f"q{qid}", ha="center", va="center",
                        fontsize=6, color="white", zorder=5, fontweight="bold")
            else:
                c = ("#1976D2" if qid in Z_ANCILLAS else "#F57C00")
                ax.plot(x, y, "D", ms=12, color="#cce0f5" if dim else c, zorder=4,
                        markeredgecolor="white", markeredgewidth=1)
                ax.text(x, y, f"q{qid}", ha="center", va="center",
                        fontsize=5, color="white" if not dim else "#aaa",
                        zorder=5, fontweight="bold")
        for ctrl, tgt in CX_LAYERS[layer_i]:
            x0, y0 = QUBIT_COORDS[ctrl]; x1, y1 = QUBIT_COORDS[tgt]
            color = "#1565C0" if ctrl in X_ANCILLAS else "#E65100"
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=1.5, mutation_scale=12,
                                        shrinkA=8, shrinkB=8), zorder=6)
        ax.set_xlim(-0.6, 3.6); ax.set_ylim(-0.6, 3.6); ax.invert_yaxis()
    fig.legend(handles=[
        Line2D([0], [0], color="#1565C0", lw=2, label="X ancilla \u2192 data"),
        Line2D([0], [0], color="#E65100", lw=2, label="data \u2192 Z ancilla"),
    ], loc="lower center", ncol=2, fontsize=9, framealpha=0.9,
               edgecolor="#ccc", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("CNOT Schedule \u2014 all 4 layers", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"CX schedule saved to {save_path}")
    return fig


def validate_error_signature(
    error: DataQubitError, shots: int = 1000, seed: int = 42,
    threshold: float = 0.95, to_json: bool = False,
) -> dict | None:
    predicted = set(predict_syndrome_changes(error))
    samples   = sample_clean_measurements(shots=shots, seed=seed, data_error=error)
    avg_rates = np.mean(detection_events_from_measurements(samples), axis=0)
    rows, all_correct = [], True
    for i, label in enumerate(MEASUREMENT_LABELS):
        rate = float(avg_rates[i])
        tf   = label in predicted
        sd   = rate >= threshold
        match = tf == sd
        if not match:
            all_correct = False
        rows.append({"stabilizer": label, "theory_flips": tf,
                     "sim_rate": round(rate, 4), "sim_detected": sd, "match": match})
    result = {"error": {"pauli": error.pauli, "qubit": error.qubit},
              "shots": shots, "threshold": threshold,
              "all_correct": all_correct, "rows": rows}
    if to_json:
        return result
    print(f"\nError signature: {error.pauli} on q{error.qubit}  ({shots} shots)")
    print(f"{'Stabilizer':<12} {'Theory':<14} {'Sim rate':<10} {'Detected':<12} Match")
    print("-" * 54)
    for r in rows:
        print(f"{r['stabilizer']:<12} {'YES' if r['theory_flips'] else 'no':<14} "
              f"{r['sim_rate']:<10.4f} {'YES' if r['sim_detected'] else 'no':<12} "
              f"{'OK' if r['match'] else 'FAIL'}")
    print("All correct:", all_correct)
    return None


def build_patch_metadata(distance: int) -> dict:
    if distance % 2 == 0 or distance < 3:
        raise ValueError(f"Distance must be odd >= 3, got {distance}.")
    n_data = distance ** 2
    n_anc  = (n_data - 1) // 2
    return {
        "distance": distance, "num_data_qubits": n_data,
        "num_z_ancillas": n_anc, "num_x_ancillas": n_anc,
        "num_total_qubits": 2 * n_data - 1, "num_stabilizers": n_data - 1,
        "code_rate": round(1 / n_data, 6), "t_correctable": (distance - 1) // 2,
    }


def print_patch_scaling_table(distances: list[int] | None = None) -> None:
    if distances is None:
        distances = [3, 5, 7, 9, 11]
    print(f"\n{'d':<5} {'data':<8} {'ancillas':<10} {'total':<8} {'t_correct':<11} code_rate")
    print("-" * 52)
    for d in distances:
        m = build_patch_metadata(d)
        print(f"{d:<5} {m['num_data_qubits']:<8} "
              f"{m['num_z_ancillas']*2:<10} {m['num_total_qubits']:<8} "
              f"{m['t_correctable']:<11} {m['code_rate']:.6f}")


def compute_detector_reliability(
    shots: int = 500, seeds: list[int] | None = None,
    data_error: DataQubitError | None = None, to_json: bool = False,
) -> dict | None:
    require_tsim()
    seeds = seeds or [0, 1, 2, 7, 42, 99, 137, 256]
    stim_text = build_clean_part1_stim_text(data_error=data_error)
    rows = []
    for seed in seeds:
        meas    = sample_measurements_from_text(stim_text, shots=shots, seed=seed)
        manual  = detection_events_from_measurements(meas)
        sampled = sample_detectors_from_text(stim_text, shots=shots, seed=seed)
        nc = min(manual.shape[1], sampled.shape[1])
        m, s = manual[:, :nc], sampled[:, :nc]
        rows.append({"seed": seed, "exact_match": bool(np.array_equal(m, s)),
                     "column_agreement": round(float(np.mean(np.all(m == s, axis=0))), 6),
                     "cell_agreement":   round(float(np.mean(m == s)), 6)})
    n_exact  = sum(r["exact_match"] for r in rows)
    overall  = float(np.mean([r["cell_agreement"] for r in rows]))
    result   = {"shots_per_seed": shots, "num_seeds": len(seeds),
                "exact_matches": n_exact,
                "overall_cell_agreement": round(overall, 6),
                "reliable": n_exact == len(seeds), "rows": rows}
    if to_json:
        return result
    print(f"\nDetector reliability ({shots} shots x {len(seeds)} seeds)")
    print(f"{'Seed':<8} {'Exact':<8} {'Col agree':<12} Cell agree")
    print("-" * 40)
    for r in rows:
        print(f"{r['seed']:<8} {'YES' if r['exact_match'] else 'NO':<8} "
              f"{r['column_agreement']:<12.6f} {r['cell_agreement']:.6f}")
    print(f"\nExact: {n_exact}/{len(seeds)}  |  Overall: {overall:.6f}")
    return None


def validation_summary_to_json(
    *, shots: int = 500, seed: int = 42,
    errors: list[DataQubitError] | None = None,
    save_path: str | Path | None = None,
) -> dict:
    errors = errors or [DataQubitError("X", 4), DataQubitError("Z", 4),
                        DataQubitError("Y", 4), DataQubitError("X", 0)]
    base   = sample_clean_measurements(shots=shots, seed=seed)
    det    = detection_events_from_measurements(base)
    result = {
        "shots": shots, "seed": seed,
        "baseline_detection_rates": format_rates_as_dict(np.mean(det, axis=0)),
        "error_signatures": [validate_error_signature(e, shots=shots, seed=seed, to_json=True)
                             for e in errors],
        "detector_reliability": compute_detector_reliability(shots=shots, to_json=True),
        "patch_scaling": {str(d): build_patch_metadata(d) for d in [3, 5, 7, 9]},
    }
    if save_path:
        Path(save_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Validation summary saved to {save_path}")
    return result


# ── original presentation helpers ─────────────────────────────────────────────
def print_patch_diagram() -> None:
    print("\nDistance-3 Rotated Surface Code Patch")
    print("=" * 45)
    print("Data: q0-q8  |  Z ancillas: q9-q12  |  X ancillas: q13-q16\n")
    print("            Z0(q9)")
    print("        q0 ---- q1")
    print("         |  X0   |  Z1(q10)")
    print("        q3 ---- q4 ---- q5")
    print("       X2|        X3    |X1(q14)")
    print("        q6 ---- q7 ---- q8")
    print("            Z2(q11)    Z3(q12)\n")
    print("Z stabilizers detect X errors  |  X stabilizers detect Z errors")


def print_stabilizer_table() -> None:
    print("\nStabilizer Table\n" + "=" * 45)
    for label in MEASUREMENT_LABELS:
        info = STABILIZERS[label]
        print(f"{label}: basis={info['basis']}, ancilla={info['ancilla']}, "
              f"data={info['data']}")


def print_single_shot_explanation(samples: np.ndarray, title: str = "Single-shot") -> None:
    parsed = parse_clean_measurements(samples)
    print(f"\n{title}\n" + "=" * 45)
    print("Round 1:", dict(zip(MEASUREMENT_LABELS, parsed["round1"][0].tolist())))
    print("Round 2:", dict(zip(MEASUREMENT_LABELS, parsed["round2"][0].tolist())))
    print("Det:    ", dict(zip(MEASUREMENT_LABELS, parsed["detection_events"][0].tolist())))
    print("A '1' = that stabilizer changed between rounds.")


def print_reference_summary(distance: int = 3, strip_noise: bool = True) -> None:
    raw = load_star_stim_text(distance=distance)
    print(f"\nSTAR reference d={distance}\n" + "=" * 60)
    print(json.dumps(summarize_reference_stim_text(raw), indent=2))
    if strip_noise:
        print("\nNoise-stripped:")
        print(json.dumps(summarize_reference_stim_text(strip_noise_from_stim_text(raw)), indent=2))


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Team STAR Part 1.")
    parser.add_argument("--mode",
                        choices=["zero-noise", "clean", "reference", "json"],
                        default="zero-noise")
    parser.add_argument("--shots",      type=int, default=500)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--distance",   type=int, default=3)
    parser.add_argument("--keep-noise", action="store_true")
    parser.add_argument("--output-dir", type=str, default="zero_noise_plots")
    parser.add_argument("--json-out",   type=str, default=None)
    args = parser.parse_args()

    if args.mode == "zero-noise":
        run_zero_noise_suite(shots=args.shots, seed=args.seed,
                             output_dir=args.output_dir)
    elif args.mode == "clean":
        print_zero_noise_report(shots=args.shots, seed=args.seed)
    elif args.mode == "reference":
        print_reference_summary(distance=args.distance,
                                strip_noise=not args.keep_noise)
    elif args.mode == "json":
        validation_summary_to_json(shots=args.shots, seed=args.seed,
                                   save_path=args.json_out)


if __name__ == "__main__":
    main()
