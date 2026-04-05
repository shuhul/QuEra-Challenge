"""
star_noise_models.py
====================
Part 1 noise sandbox.

Wraps the frozen Part 1 clean circuit with simple, parameterized
circuit-level noise channels. This module only imports FROM
STAR_P1_Emiliano — it never modifies it.

Noise model (circuit-level, injected throughout syndrome rounds):
  p_data  — DEPOLARIZE1 on all data qubits after every TICK inside
             each syndrome round (sitter / idle noise proxy)
  p_cx    — DEPOLARIZE2 on every CX pair immediately after each CX
             layer TICK (mover / gate noise proxy)
  p_meas  — X_ERROR on Z ancillas / Z_ERROR on X ancillas immediately
             before the measurement lines (readout noise proxy)

Public API (four functions, no decoder logic):
  build_noisy_part1_stim_text(...)  -> str
  compare_clean_vs_noisy(...)       -> dict
  sweep_noise_parameter(...)        -> dict
  error_sensitivity_map(...)        -> dict
"""

from __future__ import annotations

from typing import Literal

import numpy as np

# ── imports from frozen baseline (read-only dependency) ───────────────────────
from .surface_code_d3 import (
    ALL_ANCILLAS,
    CX_LAYERS,
    DATA_QUBITS,
    MEASUREMENT_LABELS,
    QUBIT_COORDS,
    X_ANCILLAS,
    Z_ANCILLAS,
    DataQubitError,
    MeasurementFlip,
    _fmt_pairs,          # internal helper — safe to reuse, not re-export
    _fmt_targets,        # same
    build_clean_part1_stim_text,
    detection_events_from_measurements,
    format_rates_as_dict,
    sample_detectors_from_text,
    sample_measurements_from_text,
)

# ── type alias ────────────────────────────────────────────────────────────────
InitState = Literal["plus", "zero"]


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════════════

def _build_noisy_syndrome_round_lines(
    p_data: float,
    p_cx: float,
    p_meas: float,
    measurement_flip: MeasurementFlip | None = None,
) -> list[str]:
    """
    Rebuild one syndrome round with circuit-level noise injected at
    every gate layer.  Structure mirrors build_syndrome_round_lines()
    from the frozen baseline exactly — noise lines are the only addition.

    Noise injection points per round
    ---------------------------------
    After reset TICK          : DEPOLARIZE1(p_data) on data qubits
    After each CX layer TICK  : DEPOLARIZE2(p_cx)   on CX pairs
                                DEPOLARIZE1(p_data)  on data qubits
                                  (idle data qubits also decohere)
    Before measurement lines  : X_ERROR(p_meas) on Z ancillas
                                Z_ERROR(p_meas) on X ancillas
    """
    lines: list[str] = [
        "# Reset ancillas",
        f"R {_fmt_targets(Z_ANCILLAS)}",
        f"RX {_fmt_targets(X_ANCILLAS)}",
        "TICK",
    ]

    # Idle data-qubit noise after reset tick
    if p_data > 0.0:
        lines.append(f"DEPOLARIZE1({p_data}) {_fmt_targets(DATA_QUBITS)}")

    lines.append("# CX layers with circuit-level noise")
    for layer in CX_LAYERS:
        lines.append(f"CX {_fmt_pairs(layer)}")
        lines.append("TICK")
        # Gate noise on every qubit touched by this CX layer
        if p_cx > 0.0:
            lines.append(f"DEPOLARIZE2({p_cx}) {_fmt_pairs(layer)}")
        # Idle sitter noise on ALL data qubits after every layer
        if p_data > 0.0:
            lines.append(f"DEPOLARIZE1({p_data}) {_fmt_targets(DATA_QUBITS)}")

    # Optional deterministic measurement flip (preserved from baseline)
    if measurement_flip is not None:
        if measurement_flip.ancilla in Z_ANCILLAS:
            lines.append(f"X {measurement_flip.ancilla}")
            lines.append("TICK")
        elif measurement_flip.ancilla in X_ANCILLAS:
            lines.append(f"Z {measurement_flip.ancilla}")
            lines.append("TICK")
        else:
            raise ValueError(
                f"Ancilla {measurement_flip.ancilla} is not a valid ancilla."
            )

    # Readout noise before measurement
    if p_meas > 0.0:
        lines.append(f"X_ERROR({p_meas}) {_fmt_targets(Z_ANCILLAS)}")
        lines.append(f"Z_ERROR({p_meas}) {_fmt_targets(X_ANCILLAS)}")

    lines.extend([
        "# Measure stabilizers",
        f"M {_fmt_targets(Z_ANCILLAS)}",
        f"MX {_fmt_targets(X_ANCILLAS)}",
        "TICK",
    ])
    return lines


# ═════════════════════════════════════════════════════════════════════════════
# Public API — Function 1
# ═════════════════════════════════════════════════════════════════════════════

def build_noisy_part1_stim_text(
    *,
    p_data: float = 0.0,
    p_cx: float = 0.0,
    p_meas: float = 0.0,
    init_state: InitState = "plus",
    data_error: DataQubitError | None = None,
    measurement_flip: MeasurementFlip | None = None,
) -> str:
    """
    Build a noisy version of the Part 1 two-round d=3 circuit.

    When all noise parameters are 0.0 this produces output that is
    functionally identical to build_clean_part1_stim_text() — no noise
    lines are emitted, so the control group is preserved exactly.

    Parameters
    ----------
    p_data  : DEPOLARIZE1 rate on data qubits at every gate layer tick.
    p_cx    : DEPOLARIZE2 rate on CX pairs immediately after each CX layer.
    p_meas  : X_ERROR / Z_ERROR rate on ancillas before measurement.
    init_state      : "plus" (|+⟩ data) or "zero" (|0⟩ data).
    data_error      : optional deterministic Pauli between rounds.
    measurement_flip: optional deterministic ancilla flip.

    Returns
    -------
    str  — valid Stim circuit text ready for tsim.Circuit().
    """
    if not (0.0 <= p_data <= 1.0):
        raise ValueError(f"p_data must be in [0, 1], got {p_data}")
    if not (0.0 <= p_cx <= 1.0):
        raise ValueError(f"p_cx must be in [0, 1], got {p_cx}")
    if not (0.0 <= p_meas <= 1.0):
        raise ValueError(f"p_meas must be in [0, 1], got {p_meas}")
    if init_state not in {"plus", "zero"}:
        raise ValueError("init_state must be 'plus' or 'zero'.")

    lines: list[str] = [
        "# Noisy d=3 Part 1 circuit — circuit-level noise sandbox",
        f"# p_data={p_data}  p_cx={p_cx}  p_meas={p_meas}",
    ]

    # Qubit coordinates (identical to clean baseline)
    for q in range(17):
        x, y = QUBIT_COORDS[q]
        lines.append(f"QUBIT_COORDS({x}, {y}) {q}")

    # Data-qubit initialisation
    if init_state == "plus":
        lines.extend([f"RX {_fmt_targets(DATA_QUBITS)}", "TICK"])
    else:
        lines.extend([f"R {_fmt_targets(DATA_QUBITS)}", "TICK"])

    # Resolve per-round measurement flips
    round1_flip = (
        measurement_flip
        if measurement_flip is not None and measurement_flip.round_index == 1
        else None
    )
    round2_flip = (
        measurement_flip
        if measurement_flip is not None and measurement_flip.round_index == 2
        else None
    )

    # Round 1
    lines.append("# === ROUND 1 ===")
    lines.extend(
        _build_noisy_syndrome_round_lines(
            p_data=p_data, p_cx=p_cx, p_meas=p_meas,
            measurement_flip=round1_flip,
        )
    )

    # Optional deterministic inter-round error (from baseline API)
    if data_error is not None:
        if data_error.qubit not in DATA_QUBITS:
            raise ValueError(
                f"data_error.qubit must be in {DATA_QUBITS}, got {data_error.qubit}"
            )
        lines.extend([
            f"# Deterministic {data_error.pauli} error between rounds",
            f"{data_error.pauli} {data_error.qubit}",
            "TICK",
        ])

    # Round 2
    lines.append("# === ROUND 2 ===")
    lines.extend(
        _build_noisy_syndrome_round_lines(
            p_data=p_data, p_cx=p_cx, p_meas=p_meas,
            measurement_flip=round2_flip,
        )
    )

    # Detector definitions — identical to clean baseline
    lines.append("# Detection events: round2 XOR round1")
    for i, label in enumerate(MEASUREMENT_LABELS):
        anc = ALL_ANCILLAS[i]
        x, y = QUBIT_COORDS[anc]
        lines.append(
            f"DETECTOR({x}, {y}, 1) rec[-{8 - i}] rec[-{16 - i}]  # {label}"
        )

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Public API — Function 2
# ═════════════════════════════════════════════════════════════════════════════

def compare_clean_vs_noisy(
    *,
    p_data: float = 0.01,
    p_cx: float = 0.0,
    p_meas: float = 0.0,
    shots: int = 500,
    seed: int = 42,
    data_error: DataQubitError | None = None,
) -> dict:
    """
    Sample both the clean and noisy circuits and return per-stabilizer
    average detection rates for immediate plotting.

    Returns
    -------
    dict with keys:
      "clean"       : dict[str, float]  — rates from frozen baseline
      "noisy"       : dict[str, float]  — rates from noisy circuit
      "labels"      : list[str]         — stabilizer label order
      "p_data"      : float
      "p_cx"        : float
      "p_meas"      : float
      "shots"       : int
      "seed"        : int
    """
    # Clean baseline — use the frozen builder directly
    clean_text = build_clean_part1_stim_text(data_error=data_error)
    clean_meas = sample_measurements_from_text(clean_text, shots=shots, seed=seed)
    clean_det  = detection_events_from_measurements(clean_meas)
    clean_rates = format_rates_as_dict(np.mean(clean_det, axis=0))

    # Noisy circuit
    noisy_text = build_noisy_part1_stim_text(
        p_data=p_data, p_cx=p_cx, p_meas=p_meas, data_error=data_error
    )
    noisy_meas = sample_measurements_from_text(noisy_text, shots=shots, seed=seed)
    noisy_det  = detection_events_from_measurements(noisy_meas)
    noisy_rates = format_rates_as_dict(np.mean(noisy_det, axis=0))

    return {
        "clean":  clean_rates,
        "noisy":  noisy_rates,
        "labels": list(MEASUREMENT_LABELS),
        "p_data": p_data,
        "p_cx":   p_cx,
        "p_meas": p_meas,
        "shots":  shots,
        "seed":   seed,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Public API — Function 3
# ═════════════════════════════════════════════════════════════════════════════

def sweep_noise_parameter(
    param: Literal["p_data", "p_cx", "p_meas"],
    values: list[float],
    *,
    shots: int = 500,
    seed: int = 42,
    data_error: DataQubitError | None = None,
) -> dict:
    """
    Sweep one noise parameter across a list of values while holding
    the other two at zero.  Returns per-stabilizer detection rates at
    each value for line-plot visualisation.

    Parameters
    ----------
    param  : which noise channel to sweep ("p_data", "p_cx", "p_meas").
    values : list of float noise rates to evaluate.
    shots  : shots per sample point.
    seed   : random seed (same across all values for comparability).

    Returns
    -------
    dict with keys:
      "param"   : str             — swept parameter name
      "values"  : list[float]     — the input noise values
      "labels"  : list[str]       — stabilizer label order
      "rates"   : list[dict]      — one dict per value, keyed by label
      "shots"   : int
      "seed"    : int
    """
    if param not in {"p_data", "p_cx", "p_meas"}:
        raise ValueError(f"param must be 'p_data', 'p_cx', or 'p_meas', got {param!r}")

    rates_per_value: list[dict[str, float]] = []

    for v in values:
        kwargs: dict = {"p_data": 0.0, "p_cx": 0.0, "p_meas": 0.0}
        kwargs[param] = v

        text = build_noisy_part1_stim_text(**kwargs, data_error=data_error)
        meas = sample_measurements_from_text(text, shots=shots, seed=seed)
        det  = detection_events_from_measurements(meas)
        rates_per_value.append(format_rates_as_dict(np.mean(det, axis=0)))

    return {
        "param":  param,
        "values": values,
        "labels": list(MEASUREMENT_LABELS),
        "rates":  rates_per_value,
        "shots":  shots,
        "seed":   seed,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Public API — Function 4
# ═════════════════════════════════════════════════════════════════════════════

def error_sensitivity_map(
    pauli: Literal["X", "Z", "Y"] = "X",
    *,
    p_data: float = 0.01,
    p_cx: float = 0.0,
    p_meas: float = 0.0,
    shots: int = 500,
    seed: int = 42,
) -> dict:
    """
    For each of the 9 data qubits, run the noisy circuit and record
    per-stabilizer detection rates.  Returns data for the spatial
    sensitivity heatmap (Figure 3).

    Returns
    -------
    dict with keys:
      "pauli"       : str
      "p_data"      : float
      "p_cx"        : float
      "p_meas"      : float
      "labels"      : list[str]          — stabilizer order
      "per_qubit"   : dict[int, dict]    — qubit -> stabilizer rates
      "total_rate"  : dict[int, float]   — qubit -> sum of all rates
      "shots"       : int
      "seed"        : int
    """
    if pauli not in {"X", "Z", "Y"}:
        raise ValueError(f"pauli must be 'X', 'Z', or 'Y', got {pauli!r}")

    per_qubit: dict[int, dict[str, float]] = {}
    total_rate: dict[int, float] = {}

    for q in DATA_QUBITS:
        error = DataQubitError(pauli=pauli, qubit=q)  # type: ignore[arg-type]
        text  = build_noisy_part1_stim_text(
            p_data=p_data, p_cx=p_cx, p_meas=p_meas, data_error=error
        )
        meas  = sample_measurements_from_text(text, shots=shots, seed=seed)
        det   = detection_events_from_measurements(meas)
        rates = format_rates_as_dict(np.mean(det, axis=0))
        per_qubit[q]  = rates
        total_rate[q] = float(sum(rates.values()))

    return {
        "pauli":      pauli,
        "p_data":     p_data,
        "p_cx":       p_cx,
        "p_meas":     p_meas,
        "labels":     list(MEASUREMENT_LABELS),
        "per_qubit":  per_qubit,
        "total_rate": total_rate,
        "shots":      shots,
        "seed":       seed,
    }