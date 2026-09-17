"""
Role 3: Noise Robustness and SNR Sensitivity Test
=================================================
Injects complex Additive White Gaussian Noise (AWGN) into the target slow-time signal
across SNR levels [30 dB, 20 dB, 10 dB, 5 dB] to quantify:
- Phase unwrapping stability & cycle slip occurrence
- Displacement reconstruction RMSE
- Respiration and cardiac BPM estimation errors
"""

import os
import json
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


def run_noise_robustness_test(
    target_bin_path="outputs/target_bin_slow_time.npy",
    config_path="config.json",
    output_dir="outputs",
    plot_dir="plots"
):
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    if not os.path.isabs(target_bin_path):
        target_bin_path = os.path.join(base_dir, "..", target_bin_path)
    if not os.path.isabs(config_path):
        config_path = os.path.join(base_dir, "..", config_path)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(base_dir, "..", output_dir)
    if not os.path.isabs(plot_dir):
        plot_dir = os.path.join(base_dir, "..", plot_dir)

    with open(config_path, "r") as f:
        config = json.load(f)

    target_bin = np.load(target_bin_path)
    num_chirps = len(target_bin)
    
    fs_slow = 1.0 / (config["fmcw"]["chirp_duration_s"] + config["fmcw"]["gap_duration_s"])
    slow_time = np.arange(num_chirps) / fs_slow
    fc = config["fmcw"]["f_start_hz"] + config["fmcw"]["bandwidth_hz"] / 2.0
    c = config["synthetic_target"]["speed_of_sound_mps"]

    # Baseline clean displacement
    clean_phase = np.unwrap(np.angle(target_bin))
    clean_disp = (c * (clean_phase - clean_phase[0])) / (4 * np.pi * fc) * 1000.0  # in mm

    snr_levels_db = [30, 20, 10, 5]
    sig_power = np.mean(np.abs(target_bin)**2)
    rng = np.random.default_rng(42)

    results_by_snr = {}
    noisy_displacements = {}

    for snr_db in snr_levels_db:
        noise_power = sig_power / (10 ** (snr_db / 10.0))
        noise_std = np.sqrt(noise_power / 2.0)
        
        noise = rng.normal(0, noise_std, size=num_chirps) + 1j * rng.normal(0, noise_std, size=num_chirps)
        noisy_target = target_bin + noise
        
        noisy_phase = np.unwrap(np.angle(noisy_target))
        noisy_disp = (c * (noisy_phase - noisy_phase[0])) / (4 * np.pi * fc) * 1000.0
        noisy_displacements[snr_db] = noisy_disp

        # RMSE
        rmse_mm = float(np.sqrt(np.mean((noisy_disp - clean_disp)**2)))

        # Respiration estimation on noisy signal
        sos_resp = signal.butter(4, [0.10, 0.50], btype="bandpass", fs=fs_slow, output="sos")
        resp_filt = signal.sosfiltfilt(sos_resp, (noisy_disp - np.mean(noisy_disp)) / 1000.0)
        
        N_fft = 2048
        f_axis = np.fft.fftfreq(N_fft, d=1/fs_slow)[:N_fft // 2]
        resp_fft = np.abs(np.fft.fft(resp_filt, n=N_fft))[:N_fft // 2]
        mask = (f_axis >= 0.10) & (f_axis <= 0.50)
        est_f = f_axis[mask][np.argmax(resp_fft[mask])]
        est_bpm = float(est_f * 60.0)

        results_by_snr[f"{snr_db}dB"] = {
            "snr_db": snr_db,
            "rmse_mm": rmse_mm,
            "estimated_resp_bpm": est_bpm,
            "resp_error_bpm": abs(est_bpm - config["synthetic_target"]["respiration_bpm"])
        }

    # Plot Noise Robustness Comparison
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(slow_time, clean_disp, label="Clean Baseline", color="black", lw=2, zorder=5)
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    for snr_db, color in zip(snr_levels_db, colors):
        ax1.plot(slow_time, noisy_displacements[snr_db], label=f"SNR = {snr_db} dB", color=color, alpha=0.75, lw=1.2)
    ax1.set_title("Displacement Waveform Under Additive Noise", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Slow Time (s)")
    ax1.set_ylabel("Displacement (mm)")
    ax1.legend(loc="upper right")

    # RMSE vs SNR curve
    snrs = [r["snr_db"] for r in results_by_snr.values()]
    rmses = [r["rmse_mm"] for r in results_by_snr.values()]
    ax2.plot(snrs, rmses, "-o", color="#d62728", lw=2, markersize=7)
    ax2.set_title("Displacement Reconstruction RMSE vs SNR", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Input SNR (dB)")
    ax2.set_ylabel("RMSE (mm)")
    ax2.set_xticks(snr_levels_db)
    ax2.invert_xaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "noise_robustness_comparison.png"), dpi=200)
    plt.close()

    with open(os.path.join(output_dir, "noise_test_results.json"), "w") as f:
        json.dump(results_by_snr, f, indent=2)

    return results_by_snr
