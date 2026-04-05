"""
Validate_STAR_P1_Emiliano.py new
===========================

Presentation-friendly runner and validator for Team STAR Part 1.

Enhancements over original:
- Calls validate_error_signature()     for structured theory vs sim tables
- Calls plot_patch()                   for matplotlib patch geometry
- Calls plot_cx_schedule()             for per-layer CNOT visualization
- Calls plot_syndrome_heatmap()        for shot-by-shot syndrome patterns
- Calls compute_detector_reliability() for cross-seed reliability scoring
- Calls build_patch_metadata()         for distance scaling
- Calls validation_summary_to_json()   for full JSON export
- All functions accept --json-out CLI flag for piping to notebooks/CI
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np

from STAR_P1_Emiliano import (
    DataQubitError,
    MeasurementFlip,
    MEASUREMENT_LABELS,
    STAR_CIRCUIT_DIR,
    build_patch_metadata,
    compute_detector_reliability,
    detection_events_from_measurements,
    format_rates_as_dict,
    list_available_star_distances,
    load_star_stim_text,
    parse_clean_measurements,
    plot_cx_schedule,
    plot_patch,
    plot_syndrome_heatmap,
    predict_syndrome_changes,
    print_patch_diagram,
    print_patch_scaling_table,
    print_reference_summary,
    print_single_shot_explanation,
    print_stabilizer_table,
    require_tsim,
    sample_clean_detectors,
    sample_clean_measurements,
    sample_reference_detectors,
    sample_reference_measurements,
    strip_noise_from_stim_text,
    validate_error_signature,
    validation_summary_to_json,
)


def print_section(title: str) -> None:
    """Pretty divider for terminal output."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ══════════════════════════════════════════════════════════════════════════════
# Clean two-round Part 1 validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_clean_case(
    title: str,
    error: DataQubitError | None = None,
    *,
    shots: int = 100,
    seed: int = 42,
    to_json: bool = False,
) -> dict | None:
    """
    Run one clean two-round case and print theory vs simulation.

    With to_json=True returns a dict instead of printing.
    """
    print_section(title)
    samples = sample_clean_measurements(shots=shots, seed=seed, data_error=error)
    parsed = parse_clean_measurements(samples)
    det = parsed["detection_events"]
    avg_rates = np.mean(det, axis=0)

    result = {
        "title": title,
        "shots": shots,
        "measurement_shape": list(samples.shape),
        "avg_detection_rates": format_rates_as_dict(avg_rates),
    }

    if error is not None:
        predicted = predict_syndrome_changes(error)
        result["error"] = {"pauli": error.pauli, "qubit": error.qubit}
        result["predicted_flips"] = predicted

    if to_json:
        return result

    print("Measurement shape:", samples.shape)
    print("Average detection-event rates:")
    print(format_rates_as_dict(avg_rates))

    if error is not None:
        print("\nExpected flipped stabilizers (theory):", predict_syndrome_changes(error))
        print("\nDetailed theory vs simulation:")
        validate_error_signature(error, shots=shots, seed=seed)

    print_single_shot_explanation(samples, title=f"{title} (single-shot)")
    return None


def validate_measurement_flip(
    *, shots: int = 100, seed: int = 42, to_json: bool = False
) -> dict | None:
    """Show what a deterministic measurement flip looks like."""
    print_section("Clean Part 1: measurement-flip example")

    samples = sample_clean_measurements(
        shots=shots, seed=seed,
        measurement_flip=MeasurementFlip(ancilla=10, round_index=2),
    )
    det = detection_events_from_measurements(samples)
    avg_rates = np.mean(det, axis=0)
    rates = format_rates_as_dict(avg_rates)

    result = {
        "flip": {"ancilla": 10, "round_index": 2, "stabilizer": "Z1"},
        "avg_detection_rates": rates,
    }

    if to_json:
        return result

    print("Inserted deterministic measurement flip on ancilla 10 (Z1) in round 2.")
    print("Average detection-event rates:")
    print(rates)
    print_single_shot_explanation(samples, title="Measurement-flip example")
    return None


def compare_manual_vs_detector_sampler(
    *, shots: int = 100, seed: int = 42,
    error: DataQubitError | None = None,
    to_json: bool = False,
) -> dict | None:
    """
    Compare manual round2 XOR round1 with detector sampler output.

    Verifies that the detector definitions in the circuit match the
    manual XOR computation across multiple seeds.
    """
    print_section("Clean Part 1: manual XOR vs detector sampler")

    seeds_to_check = [seed, seed + 1, seed + 7, seed + 42]
    all_match = True
    seed_results = []

    for s in seeds_to_check:
        meas = sample_clean_measurements(shots=shots, seed=s, data_error=error)
        manual_det = detection_events_from_measurements(meas)
        sampled_det = sample_clean_detectors(shots=shots, seed=s, data_error=error)

        n_cols = min(manual_det.shape[1], sampled_det.shape[1])
        exact = bool(np.array_equal(manual_det[:, :n_cols], sampled_det[:, :n_cols]))
        if not exact:
            all_match = False
        seed_results.append({"seed": s, "exact": exact})

    result = {
        "shots": shots,
        "seeds_checked": seeds_to_check,
        "all_exact_match": all_match,
        "per_seed": seed_results,
    }

    if to_json:
        return result

    print(f"Checking {len(seeds_to_check)} seeds with {shots} shots each.")
    for r in seed_results:
        mark = "✓" if r["exact"] else "✗"
        print(f"  seed={r['seed']:>4}  exact match: {mark}")
    print()
    if all_match:
        print("All seeds: detector sampler agrees exactly with round2 XOR round1.")
    else:
        print("WARNING: Mismatch found — inspect detector ordering or circuit semantics.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Enhanced: syndrome heatmaps for multiple error types
# ══════════════════════════════════════════════════════════════════════════════

def run_heatmap_comparison(
    shots: int = 300,
    seed: int = 42,
    save_dir: Path | None = None,
) -> None:
    """
    Generate syndrome heatmaps for no-error, X, Z, and Y on q4.

    Saves four .png files if save_dir is provided.
    """
    print_section("Syndrome heatmap comparison")

    cases = [
        (None,                   "No error"),
        (DataQubitError("X", 4), "X error on q4"),
        (DataQubitError("Z", 4), "Z error on q4"),
        (DataQubitError("Y", 4), "Y error on q4"),
    ]

    for error, label in cases:
        samples = sample_clean_measurements(shots=shots, seed=seed, data_error=error)
        det = detection_events_from_measurements(samples)
        avg = np.mean(det, axis=0)
        print(f"\n{label}:")
        print("  Avg rates:", format_rates_as_dict(avg))

        if save_dir is not None:
            fname = label.lower().replace(" ", "_").replace("/", "_") + "_heatmap.png"
            plot_syndrome_heatmap(
                samples,
                title=label,
                save_path=save_dir / fname,
                max_shots_shown=min(shots, 200),
            )


# ══════════════════════════════════════════════════════════════════════════════
# Official STAR .stim validation (preserved + enhanced)
# ══════════════════════════════════════════════════════════════════════════════

def validate_reference_file(
    distance: int = 3,
    *,
    shots: int = 100,
    seed: int = 42,
    strip_noise: bool = True,
    to_json: bool = False,
) -> dict | None:
    """Load, summarize, and sample an official STAR .stim file."""
    print_section(f"Official STAR reference validation: d={distance}")

    raw_text = load_star_stim_text(distance=distance)
    used_text = strip_noise_from_stim_text(raw_text) if strip_noise else raw_text

    print_reference_summary(distance=distance, strip_noise=strip_noise)

    measurements = sample_reference_measurements(
        distance=distance, shots=shots, seed=seed, strip_noise=strip_noise,
    )
    detectors = sample_reference_detectors(
        distance=distance, shots=shots, seed=seed, strip_noise=strip_noise,
    )

    avg_det = np.mean(detectors, axis=0)
    result = {
        "distance": distance,
        "noise_stripped": strip_noise,
        "raw_text_length": len(raw_text),
        "used_text_length": len(used_text),
        "measurement_shape": list(measurements.shape),
        "detector_shape": list(detectors.shape),
        "avg_detector_rates_first16": avg_det[:min(16, avg_det.shape[0])].tolist(),
    }

    if to_json:
        return result

    print(f"\nNoise stripped: {strip_noise}")
    print(f"Raw text length: {len(raw_text)} chars")
    print(f"Used text length: {len(used_text)} chars")
    print(f"Measurement shape: {measurements.shape}")
    print(f"Detector shape: {detectors.shape}")
    print("Avg detector firing rates (first 16):")
    print(avg_det[:min(16, avg_det.shape[0])])
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Distance scaling summary
# ══════════════════════════════════════════════════════════════════════════════

def run_scaling_summary(to_json: bool = False) -> dict | None:
    """Print or return patch scaling metadata for d=3,5,7,9,11."""
    print_section("Patch scaling summary")
    distances = [3, 5, 7, 9, 11]
    scaling = {str(d): build_patch_metadata(d) for d in distances}

    if to_json:
        return scaling

    print_patch_scaling_table(distances)
    available = list_available_star_distances()
    if available:
        print(f"\nOfficial .stim files available for d = {available}")
    else:
        print("\nNo official .stim files found in assets/star_circuits/")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Full validation run
# ══════════════════════════════════════════════════════════════════════════════

def run_validation(
    shots: int = 500,
    seed: int = 42,
    save_plots: bool = False,
    json_out: str | None = None,
) -> None:
    """Run the full Team STAR Part 1 validation suite."""
    require_tsim()

    print_section("TEAM STAR PART 1 — FULL VALIDATION")
    print("Goal:")
    print("  1. Understand the d=3 rotated surface code geometry")
    print("  2. Validate syndrome extraction against theory")
    print("  3. Cross-check detector sampler vs manual XOR")
    print("  4. Verify error signature predictions")
    print("  5. Benchmark detector reliability across seeds")
    print("  6. Inspect syndrome patterns via heatmaps")
    print("  7. Summarize patch scaling to larger distances")

    # ── static reference output ───────────────────────────────────────────────
    print_patch_diagram()
    print_stabilizer_table()

    # ── plots ─────────────────────────────────────────────────────────────────
    if save_plots:
        plot_dir = Path("plots")
        plot_dir.mkdir(exist_ok=True)
        print_section("Generating plots")
        plot_patch(save_path=plot_dir / "patch.png")
        plot_patch(
            save_path=plot_dir / "patch_X_q4.png",
            highlight_error=DataQubitError("X", 4),
        )
        plot_cx_schedule(save_path=plot_dir / "cx_schedule.png")
        run_heatmap_comparison(shots=shots, seed=seed, save_dir=plot_dir)
        print(f"All plots saved to {plot_dir}/")

    # ── clean baseline ────────────────────────────────────────────────────────
    validate_clean_case("Baseline — no error", shots=shots, seed=seed)

    # ── error signature validation for multiple error types ───────────────────
    errors_to_test = [
        DataQubitError("X", 4),
        DataQubitError("Z", 4),
        DataQubitError("Y", 4),
        DataQubitError("X", 0),
        DataQubitError("X", 8),
        DataQubitError("Z", 0),
    ]

    print_section("Error signature validation (theory vs simulation)")
    for err in errors_to_test:
        validate_error_signature(err, shots=shots, seed=seed)

    # ── measurement flip demo ─────────────────────────────────────────────────
    validate_measurement_flip(shots=shots, seed=seed)

    # ── manual XOR vs detector sampler ───────────────────────────────────────
    compare_manual_vs_detector_sampler(shots=shots, seed=seed)
    compare_manual_vs_detector_sampler(
        shots=shots, seed=seed, error=DataQubitError("X", 4)
    )

    # ── detector reliability across seeds ────────────────────────────────────
    print_section("Detector reliability (cross-seed)")
    compute_detector_reliability(shots=shots)
    print("\nWith X error on q4:")
    compute_detector_reliability(shots=shots, data_error=DataQubitError("X", 4))

    # ── scaling summary ───────────────────────────────────────────────────────
    run_scaling_summary()

    # ── JSON export ───────────────────────────────────────────────────────────
    if json_out is not None:
        print_section("Exporting JSON summary")
        result = validation_summary_to_json(
            shots=shots,
            seed=seed,
            errors=errors_to_test,
            save_path=json_out,
        )
        print(f"JSON written to {json_out}")
    else:
        # always print a compact inline JSON of the key results
        print_section("Compact JSON summary (stdout)")
        result = validation_summary_to_json(shots=shots, seed=seed)
        compact = {
            "baseline_detection_rates": result["baseline_detection_rates"],
            "detector_reliability": {
                k: v for k, v in result["detector_reliability"].items()
                if k != "rows"
            },
        }
        print(json.dumps(compact, indent=2))

    print_section("Validation complete")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Team STAR Part 1 — enhanced validation runner."
    )
    parser.add_argument("--shots", type=int, default=500,
                        help="Shots per sampling call (default 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default 42)")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save matplotlib plots to plots/ directory")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Path to write full validation JSON output")
    parser.add_argument("--reference", action="store_true",
                        help="Also validate official STAR .stim reference files")
    parser.add_argument("--distance", type=int, default=3,
                        help="Distance for reference .stim validation (default 3)")
    args = parser.parse_args()

    run_validation(
        shots=args.shots,
        seed=args.seed,
        save_plots=args.save_plots,
        json_out=args.json_out,
    )

    if args.reference:
        validate_reference_file(
            distance=args.distance,
            shots=args.shots,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
