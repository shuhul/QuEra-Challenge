"""
surface_code_generator.py
=========================
Generates rotated surface code geometry for arbitrary odd distance d.

For distance d:
  - d² data qubits
  - (d²-1)/2 Z stabilizers
  - (d²-1)/2 X stabilizers
  - Total qubits: d² + (d²-1) = 2d² - 1

Public API:
  generate_surface_code(d) -> SurfaceCodeGeometry
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Stabilizer:
    """One stabilizer with its type, ancilla index, and data qubit indices."""
    label: str
    stab_type: Literal["Z", "X"]
    ancilla: int
    data: tuple[int, ...]


@dataclass
class SurfaceCodeGeometry:
    """Complete geometry for a distance-d rotated surface code."""
    distance: int
    data_qubits: tuple[int, ...]
    z_ancillas: tuple[int, ...]
    x_ancillas: tuple[int, ...]
    all_ancillas: tuple[int, ...]
    qubit_coords: dict[int, tuple[float, float]]
    stabilizers: dict[str, Stabilizer]
    measurement_labels: tuple[str, ...]
    cx_layers: tuple[list[tuple[int, int]], ...]
    
    @property
    def num_data(self) -> int:
        return len(self.data_qubits)
    
    @property
    def num_ancillas(self) -> int:
        return len(self.all_ancillas)
    
    @property
    def num_qubits(self) -> int:
        return self.num_data + self.num_ancillas


def generate_surface_code(d: int) -> SurfaceCodeGeometry:
    """
    Generate complete geometry for a distance-d rotated surface code.
    
    Parameters
    ----------
    d : int
        Code distance (must be odd >= 3)
    
    Returns
    -------
    SurfaceCodeGeometry with all qubit indices, coordinates, stabilizers, and CX schedule.
    """
    if d < 3 or d % 2 == 0:
        raise ValueError(f"Distance must be odd >= 3, got {d}")
    
    # ── Coordinate system ─────────────────────────────────────────────────────
    # Data qubits on integer grid (0..d-1) x (0..d-1)
    # Ancillas at half-integer positions between data qubits
    
    qubit_coords: dict[int, tuple[float, float]] = {}
    
    # Data qubits: indices 0 to d²-1
    data_qubits: list[int] = []
    q = 0
    for row in range(d):
        for col in range(d):
            data_qubits.append(q)
            qubit_coords[q] = (col, row)
            q += 1
    
    # Helper to get data qubit index from (col, row)
    def data_idx(col: int, row: int) -> int:
        return row * d + col
    
    # ── Ancillas and stabilizers ──────────────────────────────────────────────
    # Z ancillas at (col+0.5, row+0.5) for checkerboard pattern
    # X ancillas at alternating positions
    
    z_ancillas: list[int] = []
    x_ancillas: list[int] = []
    stabilizers: dict[str, Stabilizer] = {}
    
    z_count = 0
    x_count = 0
    
    # Interior ancillas (4-body stabilizers)
    for row in range(d - 1):
        for col in range(d - 1):
            # Checkerboard: (row + col) even -> Z, on top
            ancilla_idx = q
            qubit_coords[q] = (col + 0.5, -0.5)
            data = (data_idx(col, 0), data_idx(col + 1, 0))
            x_ancillas.append(q)
            label = f"X{x_count}"
            stabilizers[label] = Stabilizer(label, "X", q, data)
            x_count += 1
            q += 1
    
    # Bottom edge (row = d - 0.5)
    for col in range(d - 1):
        if ((d - 1) + col) % 2 == 1:  # X stabilizers on bottom
            ancilla_idx = q
            qubit_coords[q] = (col + 0.5, d - 0.5)
            data = (data_idx(col, d - 1), data_idx(col + 1, d - 1))
            x_ancillas.append(q)
            label = f"X{x_count}"
            stabilizers[label] = Stabilizer(label, "X", q, data)
            x_count += 1
            q += 1
    
    # Left edge (col = -0.5)
    for row in range(d - 1):
        if (row + 0) % 2 == 0:  # Z stabilizers on left
            ancilla_idx = q
            qubit_coords[q] = (-0.5, row + 0.5)
            data = (data_idx(0, row), data_idx(0, row + 1))
            z_ancillas.append(q)
            label = f"Z{z_count}"
            stabilizers[label] = Stabilizer(label, "Z", q, data)
            z_count += 1
            q += 1
    
    # Right edge (col = d - 0.5)
    for row in range(d - 1):
        if (row + (d - 1)) % 2 == 0:  # Z stabilizers on right
            ancilla_idx = q
            qubit_coords[q] = (d - 0.5, row + 0.5)
            data = (data_idx(d - 1, row), data_idx(d - 1, row + 1))
            z_ancillas.append(q)
            label = f"Z{z_count}"
            stabilizers[label] = Stabilizer(label, "Z", q, data)
            z_count += 1
            q += 1
    
    # ── CX schedule (4 layers) ────────────────────────────────────────────────
    # Standard order: NW, NE, SW, SE for each ancilla
    
    def get_cx_pairs(direction: str) -> list[tuple[int, int]]:
        """Get CX pairs for one direction (ancilla, data)."""
        pairs = []
        for label, stab in stabilizers.items():
            data_list = list(stab.data)
            ancilla = stab.ancilla
            ax, ay = qubit_coords[ancilla]
            
            # Sort data qubits by position relative to ancilla
            for dq in data_list:
                dx, dy = qubit_coords[dq]
                rel_x = "W" if dx < ax else "E"
                rel_y = "N" if dy < ay else "S"
                pos = rel_y + rel_x
                if pos == direction:
                    pairs.append((ancilla, dq))
        return pairs
    
    cx_layers = (
        get_cx_pairs("NW"),
        get_cx_pairs("NE"),
        get_cx_pairs("SW"),
        get_cx_pairs("SE"),
    )
    
    # ── Measurement labels (sorted) ───────────────────────────────────────────
    z_labels = sorted([l for l in stabilizers if l.startswith("Z")], key=lambda x: int(x[1:]))
    x_labels = sorted([l for l in stabilizers if l.startswith("X")], key=lambda x: int(x[1:]))
    measurement_labels = tuple(z_labels + x_labels)
    
    all_ancillas = tuple(z_ancillas) + tuple(x_ancillas)
    
    return SurfaceCodeGeometry(
        distance=d,
        data_qubits=tuple(data_qubits),
        z_ancillas=tuple(z_ancillas),
        x_ancillas=tuple(x_ancillas),
        all_ancillas=all_ancillas,
        qubit_coords=qubit_coords,
        stabilizers=stabilizers,
        measurement_labels=measurement_labels,
        cx_layers=cx_layers,
    )


# ── Quick validation ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    for d in [3, 5, 7]:
        geo = generate_surface_code(d)
        print(f"d={d}: {geo.num_data} data, {len(geo.z_ancillas)} Z, {len(geo.x_ancillas)} X, {geo.num_qubits} total")
        print(f"      Stabilizers: {list(geo.stabilizers.keys())[:6]}...")
        print()
