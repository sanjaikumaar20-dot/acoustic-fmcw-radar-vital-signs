"""
ROLE 3: Vital Sign Processing Module
====================================
This module processes the slow-time target-bin complex signal:
1. Load target_bin_slow_time.npy and config.json
2. Extract phase using: phase = np.angle(target_bin_slow_time)
3. Unwrap the phase using 1D unwrapping (np.unwrap)
4. Convert phase variation into displacement:
       Delta_d = c * Delta_phi / (4 * pi * f_carrier)
   where c = 343 m/s, f_carrier = 19500 Hz (center frequency)
5. Remove DC offset / linear drift via detrending
6. Apply 4th-order zero-phase Butterworth bandpass filter for respiration:
       0.1 Hz to 0.5 Hz (6 BPM to 30 BPM)
7. Apply 4th-order zero-phase Butterworth bandpass filter for cardiac:
       0.8 Hz to 2.0 Hz (48 BPM to 120 BPM)
8. Estimate respiration frequency and convert to BPM:
       respiration_BPM = respiration_frequency * 60
9. Estimate cardiac frequency and convert to BPM:
       heartbeat_BPM = heartbeat_frequency * 60
10. Save displacement_waveform.npy and results.json
11. Generate verification plots for Role 3:
       target_phase.png, displacement.png,
       respiration_spectrum.png, cardiac_spectrum.png
"""

import os
import json
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


def process_vital_signs(
    target_bin_path="outputs/target_bin_slow_time.npy",
    config_path="config.json",
    output_dir="outputs",
    plot_dir="plots",
    verbose=True
):
    """
    Executes Role 3 Vital Sign Signal Processing.
    
    Parameters:
    -----------
    target_bin_path : str
        Path to target_bin_slow_time.npy from Role 2.
    config_path : str
        Path to config.json.
    output_dir : str
        Directory to save results.json and displacement_waveform.npy.
    plot_dir : str
        Directory to save visualization plots.
    verbose : bool
        Whether to print detailed estimation metrics.
        
    Returns:
    --------
    dict : Dictionary containing vital sign estimates, errors, and validation flags.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

    if not os.path.isabs(target_bin_path):
        target_bin_path = os.path.join(base_dir, target_bin_path)
    if not os.path.isabs(config_path):
        config_path = os.path.join(base_dir, config_path)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(base_dir, output_dir)
    if not os.path.isabs(plot_dir):
        plot_dir = os.path.join(base_dir, plot_dir)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(target_bin_path):
        raise FileNotFoundError(f"Target bin slow-time file not found: {target_bin_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Simulation and radar constants
    fs = int(config["role1_to_role2_contract"]["sample_rate_hz"])
    f_start = float(config["fmcw"]["f_start_hz"])
    f_end = float(config["fmcw"]["f_end_hz"])
    B = float(config["fmcw"]["bandwidth_hz"])
    Tc = float(config["fmcw"]["chirp_duration_s"])
    Tgap = float(config["fmcw"]["gap_duration_s"])
    c = float(config["synthetic_target"]["speed_of_sound_mps"])
    
    # Ground truth reference values
    ref_resp_hz = float(config["synthetic_target"].get("respiration_hz", 0.25))
    ref_resp_bpm = float(config["synthetic_target"].get("respiration_bpm", 15.0))
    ref_card_hz = float(config["synthetic_target"].get("heartbeat_hz", 1.20))
    ref_card_bpm = float(config["synthetic_target"].get("heartbeat_bpm", 72.0))
    ref_resp_disp_mm = float(config["synthetic_target"].get("respiration_displacement_mm", 4.0))
    ref_card_disp_mm = float(config["synthetic_target"].get("heartbeat_displacement_mm", 0.15))

    # Slow-time sampling parameters
    T_frame = Tc + Tgap
    fs_slow = 1.0 / T_frame  # Typically 6.6667 Hz
    
    # Carrier frequency (center of sweep)
    f_carrier = f_start + B / 2.0  # 19500 Hz
    wavelength = c / f_carrier     # ~ 0.01759 m (17.59 mm)

    # ----------------------------------------------------
    # Step 1: Load Target Bin Slow-Time Signal
    # ----------------------------------------------------
    target_bin_slow_time = np.load(target_bin_path)
    num_chirps = len(target_bin_slow_time)
    slow_time = np.arange(num_chirps) * T_frame

    if verbose:
        print("=" * 65)
        print("ROLE 3: VITAL SIGN EXTRACTION & DSP")
        print("=" * 65)
        print(f"Loaded {num_chirps} slow-time chirps over {slow_time[-1]:.2f} s")
        print(f"Slow-time sampling rate fs_slow: {fs_slow:.4f} Hz (Nyquist: {fs_slow/2:.4f} Hz)")
        print(f"Carrier frequency f_carrier: {f_carrier:.1f} Hz (Wavelength lambda: {wavelength*1e3:.2f} mm)")

    # ----------------------------------------------------
    # Step 2: Extract & Unwrap Phase
    # ----------------------------------------------------
    # Raw wrapped phase in [-pi, pi]
    raw_phase = np.angle(target_bin_slow_time)
    
    # 1D Phase unwrapping
    unwrapped_phase = np.unwrap(raw_phase)

    # ----------------------------------------------------
    # Step 3: Phase-to-Displacement Conversion
    # ----------------------------------------------------
    # FMCW phase displacement formula:
    # Delta_phi = 4 * pi * f_carrier * Delta_d / c
    # Therefore: Delta_d = c * Delta_phi / (4 * pi * f_carrier)
    phase_delta = unwrapped_phase - unwrapped_phase[0]
    displacement = (c * phase_delta) / (4 * np.pi * f_carrier)  # In meters

    # ----------------------------------------------------
    # Step 4: Detrending / DC Removal
    # ----------------------------------------------------
    poly = np.polyfit(slow_time, displacement, deg=1)
    displacement_detrend = displacement - np.polyval(poly, slow_time)

    # ----------------------------------------------------
    # Step 5: Respiration Bandpass Filtering (0.1 Hz - 0.5 Hz)
    # ----------------------------------------------------
    sos_resp = signal.butter(4, [0.10, 0.50], btype="bandpass", fs=fs_slow, output="sos")
    resp_waveform = signal.sosfiltfilt(sos_resp, displacement_detrend)

    # ----------------------------------------------------
    # Step 6: Cardiac Bandpass Filtering (0.8 Hz - 2.0 Hz)
    # ----------------------------------------------------
    sos_card = signal.butter(4, [0.80, 2.00], btype="bandpass", fs=fs_slow, output="sos")
    card_waveform = signal.sosfiltfilt(sos_card, displacement_detrend)

    # ----------------------------------------------------
    # Step 7: Respiration Frequency & BPM Estimation
    # ----------------------------------------------------
    N_slow_fft = 4096
    f_slow_axis = np.fft.fftfreq(N_slow_fft, d=1/fs_slow)[:N_slow_fft // 2]
    
    window_resp = np.hanning(num_chirps)
    resp_fft = np.abs(np.fft.fft((resp_waveform - np.mean(resp_waveform)) * window_resp, n=N_slow_fft))[:N_slow_fft // 2]
    
    resp_mask = (f_slow_axis >= 0.10) & (f_slow_axis <= 0.50)
    resp_peak_idx = np.where(resp_mask)[0][np.argmax(resp_fft[resp_mask])]
    est_resp_hz = float(f_slow_axis[resp_peak_idx])
    
    # Time-domain zero-crossing refinement for short windows (9s = 2.25 cycles)
    zc_indices = np.where(np.diff(np.sign(resp_waveform)))[0]
    if len(zc_indices) >= 2:
        t_zc = slow_time[zc_indices]
        half_periods = np.diff(t_zc)
        mean_half_period = np.mean(half_periods)
        if mean_half_period > 0:
            est_resp_td_hz = 1.0 / (2.0 * mean_half_period)
        else:
            est_resp_td_hz = est_resp_hz
    else:
        est_resp_td_hz = est_resp_hz

    if abs(est_resp_td_hz - est_resp_hz) < 0.08:
        est_resp_final_hz = float(est_resp_td_hz)
    else:
        est_resp_final_hz = float(est_resp_hz)
    
    est_resp_bpm = float(est_resp_final_hz * 60.0)
    resp_error_bpm = abs(est_resp_bpm - ref_resp_bpm)
    est_resp_amp_mm = float(0.5 * (np.max(resp_waveform) - np.min(resp_waveform)) * 1e3)

    # ----------------------------------------------------
    # Step 8: Cardiac Frequency & BPM Estimation
    # ----------------------------------------------------
    window_card = np.hanning(num_chirps)
    card_fft = np.abs(np.fft.fft((card_waveform - np.mean(card_waveform)) * window_card, n=N_slow_fft))[:N_slow_fft // 2]
    
    card_mask = (f_slow_axis >= 0.80) & (f_slow_axis <= 2.00)
    card_peak_idx = np.where(card_mask)[0][np.argmax(card_fft[card_mask])]
    est_card_hz = float(f_slow_axis[card_peak_idx])
    est_card_bpm = float(est_card_hz * 60.0)
    card_error_bpm = abs(est_card_bpm - ref_card_bpm)
    est_card_amp_mm = float(0.5 * (np.max(card_waveform) - np.min(card_waveform)) * 1e3)

    # ----------------------------------------------------
    # Step 9: Check Signal Quality & Delay Quantization Impact
    # ----------------------------------------------------
    phase_diffs = np.abs(np.diff(unwrapped_phase))
    large_jumps = np.sum(phase_diffs > 1.5)
    sample_rounding_detected = bool(large_jumps > 0)

    cardiac_snr = np.max(card_fft[card_mask]) / (np.median(card_fft[card_mask]) + 1e-12)
    cardiac_recovered = bool((card_error_bpm < 5.0) and (cardiac_snr > 2.0))

    if verbose:
        print(f"-> Respiration Estimated: {est_resp_final_hz:.4f} Hz -> {est_resp_bpm:.2f} BPM (Expected: {ref_resp_bpm:.1f} BPM, Error: {resp_error_bpm:.2f} BPM)")
        print(f"-> Respiration Amplitude: {est_resp_amp_mm:.3f} mm (Expected: {ref_resp_disp_mm:.2f} mm)")
        print(f"-> Cardiac Estimated:     {est_card_hz:.4f} Hz -> {est_card_bpm:.2f} BPM (Expected: {ref_card_bpm:.1f} BPM, Error: {card_error_bpm:.2f} BPM)")
        print(f"-> Cardiac Amplitude:     {est_card_amp_mm:.3f} mm (Expected: {ref_card_disp_mm:.2f} mm)")
        print(f"-> Sample-delay quantization artifact detected: {sample_rounding_detected} ({large_jumps} phase jumps > 1.5 rad)")
        print(f"-> Cardiac signal recovered status: {'SUCCESS' if cardiac_recovered else 'DEGRADED / STEP-NOISE'}")

    # ----------------------------------------------------
    # Step 10: Save Outputs and Results
    # ----------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "displacement_waveform.npy"), displacement_detrend)

    results = {
        "respiration": {
            "estimated_hz": est_resp_final_hz,
            "estimated_bpm": est_resp_bpm,
            "reference_bpm": ref_resp_bpm,
            "error_bpm": resp_error_bpm,
            "estimated_amplitude_mm": est_resp_amp_mm,
            "reference_amplitude_mm": ref_resp_disp_mm,
            "is_valid": bool(resp_error_bpm < 3.0)
        },
        "cardiac": {
            "estimated_hz": est_card_hz,
            "estimated_bpm": est_card_bpm,
            "reference_bpm": ref_card_bpm,
            "error_bpm": card_error_bpm,
            "estimated_amplitude_mm": est_card_amp_mm,
            "reference_amplitude_mm": ref_card_disp_mm,
            "is_recovered": cardiac_recovered,
            "snr_metric": float(cardiac_snr)
        },
        "dataset_diagnostics": {
            "sample_delay_quantization_detected": sample_rounding_detected,
            "number_of_abrupt_phase_steps": int(large_jumps),
            "step_size_theoretical_mm": float((c / (2 * fs)) * 1e3),
            "total_chirps": num_chirps,
            "total_duration_s": float(slow_time[-1]),
            "fs_slow_hz": float(fs_slow)
        }
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"-> Saved {os.path.join(output_dir, 'displacement_waveform.npy')}")
        print(f"-> Saved {os.path.join(output_dir, 'results.json')}")

    # ----------------------------------------------------
    # Step 11: Generate Visual Verification Plots
    # ----------------------------------------------------
    os.makedirs(plot_dir, exist_ok=True)
    _generate_role3_plots(
        slow_time, raw_phase, unwrapped_phase, displacement_detrend,
        resp_waveform, card_waveform, f_slow_axis, resp_fft, card_fft,
        est_resp_final_hz, est_resp_bpm, ref_resp_bpm,
        est_card_hz, est_card_bpm, ref_card_bpm, plot_dir
    )

    if verbose:
        print(f"-> Generated Role 3 verification plots in: {plot_dir}")

    return results


def _generate_role3_plots(
    slow_time, raw_phase, unwrapped_phase, displacement_detrend,
    resp_waveform, card_waveform, f_slow_axis, resp_fft, card_fft,
    est_resp_hz, est_resp_bpm, ref_resp_bpm,
    est_card_hz, est_card_bpm, ref_card_bpm, plot_dir
):
    """Generates and saves the 4 Role-3 diagnostic plots."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Target Phase Plot (Raw vs Unwrapped)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax1.plot(slow_time, raw_phase, "-o", color="#9467bd", lw=1.5, markersize=4)
    ax1.set_title("Wrapped Phase theta[m] = angle(y[m]) vs Slow Time", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Phase (rad)")
    ax1.set_ylim(-np.pi - 0.2, np.pi + 0.2)

    ax2.plot(slow_time, unwrapped_phase, "-s", color="#1f77b4", lw=1.5, markersize=4)
    ax2.set_title("Unwrapped Continuous Phase phi[m] = unwrap(theta[m])", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Phase (rad)")
    ax2.set_xlabel("Slow Time (s)")
    ax2.set_xlim(0, slow_time[-1])
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "target_phase.png"), dpi=200)
    plt.close()

    # 2. Chest Displacement Waveform
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(slow_time, displacement_detrend * 1000, "-o", color="#d62728", lw=1.8, markersize=4, label="Total Displacement Delta d(t)")
    ax.set_title("Extracted Chest Displacement Waveform [Delta d = c * Delta phi / (4*pi*fc)]", fontsize=12, fontweight="bold")
    ax.set_xlabel("Slow Time (s)")
    ax.set_ylabel("Displacement (mm)")
    ax.set_xlim(0, slow_time[-1])
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "displacement.png"), dpi=200)
    plt.close()

    # 3. Respiration Waveform & Spectrum
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(slow_time, resp_waveform * 1000, color="#17becf", lw=2)
    ax1.set_title("Filtered Respiration Waveform (0.1 - 0.5 Hz)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Slow Time (s)")
    ax1.set_ylabel("Displacement (mm)")
    ax1.set_xlim(0, slow_time[-1])

    resp_plot_mask = (f_slow_axis >= 0.05) & (f_slow_axis <= 0.60)
    ax2.plot(f_slow_axis[resp_plot_mask] * 60, resp_fft[resp_plot_mask], color="#17becf", lw=2)
    ax2.axvline(est_resp_bpm, color="red", ls="--", lw=1.5, label=f"Estimated: {est_resp_bpm:.1f} BPM ({est_resp_hz:.3f} Hz)")
    ax2.axvline(ref_resp_bpm, color="green", ls=":", lw=1.5, label=f"Expected: {ref_resp_bpm:.1f} BPM")
    ax2.set_title("Respiration Frequency Spectrum (BPM)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Breaths Per Minute (BPM)")
    ax2.set_ylabel("Spectral Magnitude")
    ax2.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "respiration_spectrum.png"), dpi=200)
    plt.close()

    # 4. Cardiac Waveform & Spectrum
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(slow_time, card_waveform * 1000, color="#e377c2", lw=1.8)
    ax1.set_title("Filtered Cardiac Micro-motion Waveform (0.8 - 2.0 Hz)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Slow Time (s)")
    ax1.set_ylabel("Displacement (mm)")
    ax1.set_xlim(0, slow_time[-1])

    card_plot_mask = (f_slow_axis >= 0.6) & (f_slow_axis <= 2.2)
    ax2.plot(f_slow_axis[card_plot_mask] * 60, card_fft[card_plot_mask], color="#e377c2", lw=2)
    ax2.axvline(est_card_bpm, color="red", ls="--", lw=1.5, label=f"Estimated: {est_card_bpm:.1f} BPM ({est_card_hz:.3f} Hz)")
    ax2.axvline(ref_card_bpm, color="green", ls=":", lw=1.5, label=f"Expected: {ref_card_bpm:.1f} BPM")
    ax2.set_title("Cardiac Frequency Spectrum (BPM)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Heart Rate (BPM)")
    ax2.set_ylabel("Spectral Magnitude")
    ax2.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "cardiac_spectrum.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    res = process_vital_signs()
    print("Role 3 standalone execution finished successfully.")
