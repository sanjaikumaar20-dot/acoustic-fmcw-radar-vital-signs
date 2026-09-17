"""
ACOUSTIC FMCW RADAR — MASTER PIPELINE EXECUTION SCRIPT
=======================================================
This script executes the entire pipeline end-to-end:
  ROLE 1 (Simulated Data Loading & Validation)
           |
           v
  ROLE 2 (FMCW Radar DSP: Dechirping, FFT, Range Profiling, Peak Detection)
           |
           v
  ROLE 3 (Vital Sign DSP: Phase Extraction, Unwrapping, Bandpass Filtering, BPM Estimation)
           |
           v
  Results Reporting, Plot Generation & Automated Verification

Usage:
------
    python run_simulation.py
"""

import os
import sys
import json
import numpy as np

# Import Role 2 and Role 3 modules
from role2_fmcw_processing import process_fmcw
from role3_vital_sign_processing import process_vital_signs
from generate_role1_improved_simulation import generate_improved_simulation


def run_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    config_path = os.path.join(base_dir, "config.json")
    rx_path = os.path.join(base_dir, "rx_audio.npy")
    tx_path = os.path.join(base_dir, "tx_signal.npy")
    output_dir = os.path.join(base_dir, "outputs")
    plot_dir = os.path.join(base_dir, "plots")

    print("\n" + "=" * 70)
    print("      ACOUSTIC FMCW RADAR — VITAL SIGN TRACKING SIMULATION")
    print("=" * 70)

    # ----------------------------------------------------
    # Stage 1: Role 1 Verification
    # ----------------------------------------------------
    print("\n[STAGE 1 / ROLE 1] Verifying Simulated Input Data...")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing required file: {config_path}")
    if not os.path.exists(rx_path):
        raise FileNotFoundError(f"Missing required file: {rx_path}")
    if not os.path.exists(tx_path):
        raise FileNotFoundError(f"Missing required file: {tx_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    rx_audio = np.load(rx_path)
    tx_signal = np.load(tx_path)

    print(f"  ✓ config.json loaded: sample_rate={config['role1_to_role2_contract']['sample_rate_hz']} Hz")
    print(f"  ✓ rx_audio.npy: shape={rx_audio.shape}, dtype={rx_audio.dtype}, duration={len(rx_audio)/48000:.2f} s")
    print(f"  ✓ tx_signal.npy: shape={tx_signal.shape}, dtype={tx_signal.dtype}")
    print("  ✓ Role 1 simulation handoff validated successfully.")

    # ----------------------------------------------------
    # Stage 2: Role 2 FMCW Radar Processing
    # ----------------------------------------------------
    print("\n[STAGE 2 / ROLE 2] Executing FMCW Radar Signal Processing...")
    role2_results = process_fmcw(
        config_path=config_path,
        rx_path=rx_path,
        tx_path=tx_path,
        output_dir=output_dir,
        plot_dir=plot_dir,
        verbose=True
    )

    # ----------------------------------------------------
    # Stage 3: Role 3 Vital Sign Processing
    # ----------------------------------------------------
    print("\n[STAGE 3 / ROLE 3] Executing Vital Sign Extraction & Spectral Analysis...")
    target_bin_path = os.path.join(output_dir, "target_bin_slow_time.npy")
    role3_results = process_vital_signs(
        target_bin_path=target_bin_path,
        config_path=config_path,
        output_dir=output_dir,
        plot_dir=plot_dir,
        verbose=True
    )

    # ----------------------------------------------------
    # Stage 4: Comparative Diagnostics (Continuous Delay Model)
    # ----------------------------------------------------
    print("\n[SCIENTIFIC BENCHMARK] Evaluating Continuous-Delay Model...")
    rx_imp, tx_imp = generate_improved_simulation(
        config_path=config_path,
        out_rx=os.path.join(output_dir, "rx_audio_improved.npy"),
        out_tx=os.path.join(output_dir, "tx_signal_improved.npy")
    )
    
    # Process improved dataset
    role2_imp = process_fmcw(
        config_path=config_path,
        rx_path=rx_imp,
        tx_path=tx_imp,
        output_dir=os.path.join(output_dir, "improved_model"),
        plot_dir=os.path.join(plot_dir, "improved_model"),
        verbose=False
    )
    role3_imp = process_vital_signs(
        target_bin_path=os.path.join(output_dir, "improved_model", "target_bin_slow_time.npy"),
        config_path=config_path,
        output_dir=os.path.join(output_dir, "improved_model"),
        plot_dir=os.path.join(plot_dir, "improved_model"),
        verbose=False
    )

    # ----------------------------------------------------
    # Final Summary Table
    # ----------------------------------------------------
    print("\n" + "=" * 70)
    print("                    FINAL SIMULATION SUMMARY REPORT")
    print("=" * 70)
    print(f"Role 1 (Acoustic Acquisition):")
    print(f"  • Synthetic RX Samples:   {len(rx_audio)} samples (48.0 kHz, 9.0 s)")
    print(f"  • FMCW Chirp Bandwidth:   {config['fmcw']['bandwidth_hz']/1e3:.1f} kHz ({config['fmcw']['f_start_hz']/1e3:.1f} - {config['fmcw']['f_end_hz']/1e3:.1f} kHz)")
    print()
    print(f"Role 2 (Radar DSP Range Detection):")
    print(f"  • Detected Target Range:  {role2_results['detected_range_m']:.4f} m (Reference: {config['synthetic_target']['nominal_range_m']:.2f} m)")
    print(f"  • Detected Beat Freq:     {role2_results['detected_beat_hz']:.2f} Hz")
    print(f"  • Range Bin Spacing:      {role2_results['range_bin_spacing_m']*1e3:.2f} mm | Resolution: {role2_results['range_resolution_m']*1e3:.2f} mm")
    print()
    print(f"Role 3 (Vital Sign Estimation - Primary Dataset):")
    print(f"  • Estimated Respiration:  {role3_results['respiration']['estimated_bpm']:.2f} BPM (Expected: {role3_results['respiration']['reference_bpm']:.1f} BPM, Error: {role3_results['respiration']['error_bpm']:.2f} BPM)")
    print(f"  • Estimated Heart Rate:   {role3_results['cardiac']['estimated_bpm']:.2f} BPM (Expected: {role3_results['cardiac']['reference_bpm']:.1f} BPM, Error: {role3_results['cardiac']['error_bpm']:.2f} BPM)")
    print(f"  • Respiration Amplitude:  {role3_results['respiration']['estimated_amplitude_mm']:.3f} mm (Expected: {role3_results['respiration']['reference_amplitude_mm']:.2f} mm)")
    print(f"  • Cardiac Amplitude:      {role3_results['cardiac']['estimated_amplitude_mm']:.3f} mm (Expected: {role3_results['cardiac']['reference_amplitude_mm']:.2f} mm)")
    print()
    print(f"Continuous Fractional-Delay Benchmark:")
    print(f"  • Respiration BPM:        {role3_imp['respiration']['estimated_bpm']:.2f} BPM (Error: {role3_imp['respiration']['error_bpm']:.2f} BPM)")
    print(f"  • Heart Rate BPM:         {role3_imp['cardiac']['estimated_bpm']:.2f} BPM (Error: {role3_imp['cardiac']['error_bpm']:.2f} BPM)")
    print()
    print(f"Generated Output Artifacts:")
    print(f"  • outputs/range_profile.npy")
    print(f"  • outputs/range_axis.npy")
    print(f"  • outputs/target_bin_slow_time.npy")
    print(f"  • outputs/displacement_waveform.npy")
    print(f"  • outputs/results.json")
    print(f"Generated Verification Plots (in plots/):")
    print(f"  • tx_chirp.png, rx_audio.png, spectrogram.png, range_profile.png")
    print(f"  • target_bin_magnitude.png, target_phase.png, displacement.png")
    print(f"  • respiration_spectrum.png, cardiac_spectrum.png")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_pipeline()
