"""
noise_models_multiscale.py
==========================
Noise models for arbitrary distance surface codes.

Uses surface_code_generator.py to build geometry, then applies
the same noise channels as STAR_P1_Noise_models.py.
"""

from __future__ import annotations
from typing import Literal
import numpy as np
import stim


from .surface_code_generator import generate_surface_code, SurfaceCodeGeometry


def _fmt_targets(qubits: tuple[int, ...] | list[int]) -> str:
    return " ".join(str(q) for q in qubits)


def _fmt_pairs(pairs: list[tuple[int, int]]) -> str:
    return " ".join(f"{a} {b}" for a, b in pairs)


def build_noisy_circuit(
    geo: SurfaceCodeGeometry,
    *,
    p_data: float = 0.0,
    p_cx: float = 0.0,
    p_meas: float = 0.0,
    init_state: Literal["plus", "zero"] = "plus",
) -> str:
    """Build noisy two-round syndrome extraction circuit for any distance."""
    
    lines = [
        f"# Noisy d={geo.distance} surface code",
        f"# p_data={p_data}  p_cx={p_cx}  p_meas={p_meas}",
    ]
    
    # Qubit coordinates
    for q in range(geo.num_qubits):
        x, y = geo.qubit_coords[q]
        lines.append(f"QUBIT_COORDS({x}, {y}) {q}")
    
    # Initialize data qubits
    if init_state == "plus":
        lines.append(f"RX {_fmt_targets(geo.data_qubits)}")
    else:
        lines.append(f"R {_fmt_targets(geo.data_qubits)}")
    lines.append("TICK")
    
    # Two syndrome rounds
    for round_idx in [1, 2]:
        lines.append(f"# === ROUND {round_idx} ===")
        
        # Reset ancillas
        lines.append(f"R {_fmt_targets(geo.z_ancillas)}")
        lines.append(f"RX {_fmt_targets(geo.x_ancillas)}")
        lines.append("TICK")
        
        if p_data > 0:
            lines.append(f"DEPOLARIZE1({p_data}) {_fmt_targets(geo.data_qubits)}")
        
        # CX layers
        for layer in geo.cx_layers:
            if layer:
                lines.append(f"CX {_fmt_pairs(layer)}")
                lines.append("TICK")
                if p_cx > 0:
                    lines.append(f"DEPOLARIZE2({p_cx}) {_fmt_pairs(layer)}")
                if p_data > 0:
                    lines.append(f"DEPOLARIZE1({p_data}) {_fmt_targets(geo.data_qubits)}")
        
        # Measurement noise
        if p_meas > 0:
            lines.append(f"X_ERROR({p_meas}) {_fmt_targets(geo.z_ancillas)}")
            lines.append(f"Z_ERROR({p_meas}) {_fmt_targets(geo.x_ancillas)}")
        
        # Measure
        lines.append(f"M {_fmt_targets(geo.z_ancillas)}")
        lines.append(f"MX {_fmt_targets(geo.x_ancillas)}")
        lines.append("TICK")
    
    # Detectors (round 2 XOR round 1)
    n_anc = geo.num_ancillas
    for i, label in enumerate(geo.measurement_labels):
        stab = geo.stabilizers[label]
        x, y = geo.qubit_coords[stab.ancilla]
        lines.append(f"DETECTOR({x}, {y}, 1) rec[-{n_anc - i}] rec[-{2*n_anc - i}]  # {label}")
    
    return "\n".join(lines)


def sample_detection_rates(
    geo: SurfaceCodeGeometry,
    *,
    p_data: float = 0.0,
    p_cx: float = 0.0,
    p_meas: float = 0.0,
    shots: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Sample circuit and return detection rates per stabilizer."""
    
    circuit_text = build_noisy_circuit(geo, p_data=p_data, p_cx=p_cx, p_meas=p_meas)
    circuit = stim.Circuit(circuit_text)
    sampler = circuit.compile_detector_sampler()
    
    # Set numpy seed for reproducibility (Stim uses numpy internally)
    np.random.seed(seed)
    
    # sample() doesn't take seed= in newer Stim versions
    samples = sampler.sample(shots)
    
    rates = {}
    for i, label in enumerate(geo.measurement_labels):
        rates[label] = float(np.mean(samples[:, i]))
    
    return rates


def compare_distances(
    distances: list[int],
    *,
    p_data: float = 0.01,
    shots: int = 1000,
    seed: int = 42,
) -> dict:
    """Compare detection rates across multiple code distances."""
    
    results = {}
    for d in distances:
        geo = generate_surface_code(d)
        rates = sample_detection_rates(geo, p_data=p_data, shots=shots, seed=seed)
        
        # Separate bulk vs boundary
        bulk_rates = [r for l, r in rates.items() if len(geo.stabilizers[l].data) == 4]
        boundary_rates = [r for l, r in rates.items() if len(geo.stabilizers[l].data) == 2]
        
        results[d] = {
            "geometry": geo,
            "rates": rates,
            "avg_bulk": np.mean(bulk_rates) if bulk_rates else 0,
            "avg_boundary": np.mean(boundary_rates) if boundary_rates else 0,
            "avg_all": np.mean(list(rates.values())),
        }
    
    return results
