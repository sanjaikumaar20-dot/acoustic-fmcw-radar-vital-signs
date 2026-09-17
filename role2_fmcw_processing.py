"""
ROLE 2: Acoustic FMCW Radar DSP Module
=======================================
This module implements the complete radar DSP pipeline:
1. Load rx_audio.npy, tx_signal.npy, and config.json
2. Segment the continuous audio stream into individual FMCW chirps
3. Generate analytic signals via Hilbert transform (scipy.signal.hilbert)
4. Perform FMCW dechirping/mixing: s_beat[n] = x_tx[n] * conj(x_rx[n])
5. Perform fast-time zero-padded FFT to generate the range profile
6. Map beat frequencies to distance: R = c * f_beat * Tc / (2 * B)
7. Detect target range peak near 1.0 m (rejecting direct leakage near 0 m)
8. Track target range bin across chirps (slow time)
9. Save range_profile.npy, range_axis.npy, target_bin_slow_time.npy
10. Generate verification plots for Role 2
"""

import os
import json
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


def process_fmcw(
    config_path="config.json",
    rx_path="rx_audio.npy",
    tx_path="tx_signal.npy",
    output_dir="outputs",
    plot_dir="plots",
    verbose=True
):
    """
    Executes Role 2 FMCW Radar DSP.
    
    Parameters:
    -----------
    config_path : str
        Path to config.json containing simulation and radar parameters.
    rx_path : str
        Path to rx_audio.npy (received acoustic signal).
    tx_path : str
        Path to tx_signal.npy (transmitted reference signal).
    output_dir : str
        Directory to save intermediate .npy output files.
    plot_dir : str
        Directory to save visualization plots.
    verbose : bool
        Whether to print detailed DSP logging.
        
    Returns:
    --------
    dict : Dictionary containing detected target parameters and metrics.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    
    if not os.path.isabs(config_path):
        config_path = os.path.join(base_dir, config_path)
    if not os.path.isabs(rx_path):
        rx_path = os.path.join(base_dir, rx_path)
    if not os.path.isabs(tx_path):
        tx_path = os.path.join(base_dir, tx_path)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(base_dir, output_dir)
    if not os.path.isabs(plot_dir):
        plot_dir = os.path.join(base_dir, plot_dir)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(rx_path):
        raise FileNotFoundError(f"RX audio file not found: {rx_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    fs = int(config["role1_to_role2_contract"]["sample_rate_hz"])
    f_start = float(config["fmcw"]["f_start_hz"])
    f_end = float(config["fmcw"]["f_end_hz"])
    B = float(config["fmcw"]["bandwidth_hz"])
    Tc = float(config["fmcw"]["chirp_duration_s"])
    Tgap = float(config["fmcw"]["gap_duration_s"])
    num_chirps = int(config["fmcw"]["num_chirps"])
    c = float(config["synthetic_target"]["speed_of_sound_mps"])
    nominal_range = float(config["synthetic_target"].get("nominal_range_m", 1.0))

    rx_audio = np.load(rx_path)

    if verbose:
        print("=" * 65)
        print("ROLE 2: FMCW RADAR SIGNAL PROCESSING")
        print("=" * 65)
        print(f"Loaded RX signal: {len(rx_audio)} samples ({len(rx_audio)/fs:.2f} s) at {fs} Hz")
        print(f"FMCW Parameters: {f_start/1e3:.1f} - {f_end/1e3:.1f} kHz | Bandwidth B: {B/1e3:.1f} kHz")
        print(f"Chirp duration Tc: {Tc*1e3:.1f} ms | Gap duration: {Tgap*1e3:.1f} ms | Chirps: {num_chirps}")
        print(f"Speed of sound c: {c} m/s | Reference range: {nominal_range:.2f} m")

    # ----------------------------------------------------
    # Step 2: Construct Reference Transmitted Chirp
    # ----------------------------------------------------
    samples_chirp = int(Tc * fs)
    samples_gap = int(Tgap * fs)
    frame_len = samples_chirp + samples_gap

    t_chirp = np.arange(samples_chirp) / fs
    K = B / Tc  # Chirp rate in Hz/s
    tx_ref = np.cos(2 * np.pi * (f_start * t_chirp + 0.5 * K * t_chirp**2))
    
    # Analytic reference chirp: x_tx(t) = exp(j * 2*pi * (f_start * t + 0.5 * K * t^2))
    tx_analytic = signal.hilbert(tx_ref)

    # ----------------------------------------------------
    # Step 3: Segment Chirps and Perform FMCW Dechirping
    # ----------------------------------------------------
    # Fast-time zero-padded FFT size across fast-time
    Nfft = samples_chirp * 4
    range_profiles = []

    for k in range(num_chirps):
        start = k * frame_len
        end = start + samples_chirp
        if end > len(rx_audio):
            raise ValueError(f"RX audio has insufficient samples for chirp {k}")

        rx_seg = rx_audio[start:end]
        
        # Analytic signal of received chirp segment
        rx_analytic = signal.hilbert(rx_seg)

        # Dechirping / Mixing:
        # s_beat(t) = x_tx(t) * conj(x_rx(t))
        # Instantaneous beat frequency is positive: f_b = +K * tau = 2*B*R / (c*Tc)
        beat = tx_analytic * np.conj(rx_analytic)

        # Windowing (Hamming) reduces sidelobes and spectral leakage
        window = np.hamming(samples_chirp)
        beat_windowed = beat * window
        
        # Fast-time FFT
        spec = np.fft.fft(beat_windowed, n=Nfft)
        
        # Retain positive beat frequency bins
        range_profiles.append(spec[:Nfft // 2])

    range_profiles = np.array(range_profiles)  # Shape: (num_chirps, Nfft//2)

    # ----------------------------------------------------
    # Step 4: Convert Beat Frequency to Distance (Range Axis)
    # ----------------------------------------------------
    # Range formula: R = c * f_beat * Tc / (2 * B)
    freq_axis = np.fft.fftfreq(Nfft, d=1/fs)[:Nfft // 2]
    range_axis = c * freq_axis * Tc / (2 * B)

    # ----------------------------------------------------
    # Step 5: Detect Target Range Peak
    # ----------------------------------------------------
    # Average magnitude profile across all chirps
    mean_profile = np.mean(np.abs(range_profiles), axis=0)

    # Direct acoustic leakage sits near range < 0.4 m (beat freq < 70 Hz)
    # Search for target peak in expected window [0.5 m, 2.0 m]
    search_mask = (range_axis >= 0.5) & (range_axis <= 2.0)
    target_idx_sub = np.argmax(mean_profile[search_mask])
    target_bin_idx = np.where(search_mask)[0][target_idx_sub]

    detected_range_m = float(range_axis[target_bin_idx])
    detected_beat_hz = float(freq_axis[target_bin_idx])

    if verbose:
        print(f"-> Detected Target Range: {detected_range_m:.4f} m (Bin: {target_bin_idx}, Beat Freq: {detected_beat_hz:.2f} Hz)")
        print(f"-> Target Range Estimation Error: {abs(detected_range_m - nominal_range)*1e3:.2f} mm")

    # ----------------------------------------------------
    # Step 6: Extract Complex Slow-Time Target Bin Signal
    # ----------------------------------------------------
    # y[m] for each chirp m = 0 ... num_chirps-1
    target_bin_slow_time = range_profiles[:, target_bin_idx]

    # ----------------------------------------------------
    # Step 7: Save Intermediate Output Arrays
    # ----------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "range_profile.npy"), range_profiles)
    np.save(os.path.join(output_dir, "range_axis.npy"), range_axis)
    np.save(os.path.join(output_dir, "target_bin_slow_time.npy"), target_bin_slow_time)

    if verbose:
        print(f"-> Saved {os.path.join(output_dir, 'range_profile.npy')}: shape {range_profiles.shape}")
        print(f"-> Saved {os.path.join(output_dir, 'range_axis.npy')}: shape {range_axis.shape}")
        print(f"-> Saved {os.path.join(output_dir, 'target_bin_slow_time.npy')}: shape {target_bin_slow_time.shape} (complex128)")

    # ----------------------------------------------------
    # Step 8: Generate Visual Verification Plots
    # ----------------------------------------------------
    os.makedirs(plot_dir, exist_ok=True)
    _generate_role2_plots(
        t_chirp, tx_ref, rx_audio, fs, f_start, f_end,
        range_axis, mean_profile, range_profiles, target_bin_idx,
        detected_range_m, detected_beat_hz, num_chirps, Tc, Tgap, plot_dir
    )

    if verbose:
        print(f"-> Generated Role 2 verification plots in: {plot_dir}")

    return {
        "detected_range_m": detected_range_m,
        "detected_beat_hz": detected_beat_hz,
        "target_bin_idx": int(target_bin_idx),
        "num_chirps": num_chirps,
        "range_profiles_shape": list(range_profiles.shape),
        "range_axis_max_m": float(range_axis[-1]),
        "range_resolution_m": float(c / (2 * B)),
        "range_bin_spacing_m": float(range_axis[1] - range_axis[0])
    }


def _generate_role2_plots(
    t_chirp, tx_ref, rx_audio, fs, f_start, f_end,
    range_axis, mean_profile, range_profiles, target_bin_idx,
    detected_range_m, detected_beat_hz, num_chirps, Tc, Tgap, plot_dir
):
    """Generates and saves the 5 Role-2 diagnostic plots."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Transmitted FMCW Chirp Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
    ax1.plot(t_chirp * 1000, tx_ref, color="#1f77b4", lw=1.2)
    ax1.set_title("Transmitted FMCW Chirp Signal s_tx(t)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("Time (ms)")
    ax1.set_xlim(0, Tc * 1000)
    
    # Zoom-in to show the frequency chirp sweep
    zoom_samples = int(0.005 * fs)
    ax2.plot(t_chirp[:zoom_samples] * 1000, tx_ref[:zoom_samples], color="#ff7f0e", lw=1.5)
    ax2.set_title(f"Zoomed-in View (First 5 ms, sweeping {f_start/1e3:.1f} kHz -> {f_end/1e3:.1f} kHz)", fontsize=10)
    ax2.set_ylabel("Amplitude")
    ax2.set_xlabel("Time (ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "tx_chirp.png"), dpi=200)
    plt.close()

    # 2. Received Audio Signal Plot
    t_rx = np.arange(len(rx_audio)) / fs
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_rx, rx_audio, color="#2ca02c", lw=0.8, alpha=0.85)
    ax.set_title("Received Acoustic FMCW Audio s_rx(t) - 60 Chirp Frames", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, t_rx[-1])
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "rx_audio.png"), dpi=200)
    plt.close()

    # 3. Time-Frequency Spectrogram
    fig, ax = plt.subplots(figsize=(10, 4.5))
    f_spec, t_spec, Sxx = signal.spectrogram(rx_audio, fs=fs, nperseg=1024, noverlap=512)
    band_mask = (f_spec >= 16000) & (f_spec <= 23000)
    pcm = ax.pcolormesh(t_spec, f_spec[band_mask] / 1000, 10 * np.log10(Sxx[band_mask, :] + 1e-12),
                         shading="gouraud", cmap="magma")
    fig.colorbar(pcm, ax=ax, label="Power Spectral Density (dB/Hz)")
    ax.set_title("Acoustic FMCW Spectrogram (Repeated 18 kHz - 21 kHz Chirps)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "spectrogram.png"), dpi=200)
    plt.close()

    # 4. Range Profile (Fast-Time FFT)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_mask = (range_axis >= 0.0) & (range_axis <= 3.0)
    ax1.plot(range_axis[plot_mask], mean_profile[plot_mask], color="#0055aa", lw=2, label="Mean Range Profile")
    ax1.axvline(detected_range_m, color="red", ls="--", lw=1.8,
                label=f"Target Peak @ {detected_range_m:.3f} m ({detected_beat_hz:.1f} Hz)")
    ax1.scatter([detected_range_m], [mean_profile[target_bin_idx]], color="red", s=60, zorder=5)
    ax1.set_title("Fast-Time FFT Range Profile [R = c * f_beat * Tc / (2B)]", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Range (m)")
    ax1.set_ylabel("Magnitude (a.u.)")
    ax1.legend(loc="upper right", frameon=True)

    # Subplot 2: 2D Range-SlowTime Heatmap
    r_map_mask = (range_axis >= 0.2) & (range_axis <= 2.5)
    r_sub = range_axis[r_map_mask]
    img = ax2.imshow(np.abs(range_profiles[:, r_map_mask]), aspect="auto",
                     extent=[r_sub[0], r_sub[-1], num_chirps, 1], cmap="viridis", origin="upper")
    ax2.axvline(detected_range_m, color="red", ls="--", lw=1.5, alpha=0.8)
    fig.colorbar(img, ax=ax2, label="Magnitude")
    ax2.set_title("Range Profile vs Chirp Index (Slow Time)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Range (m)")
    ax2.set_ylabel("Chirp Number")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "range_profile.png"), dpi=200)
    plt.close()

    # 5. Target Bin Magnitude Across Slow-Time
    target_mag = np.abs(range_profiles[:, target_bin_idx])
    slow_time = np.arange(num_chirps) * (Tc + Tgap)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(slow_time, target_mag, "-o", color="#8c564b", lw=1.8, markersize=4)
    ax.set_title(f"Target Range Bin Magnitude vs Slow Time (Range = {detected_range_m:.3f} m)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Slow Time (s)")
    ax.set_ylabel("Magnitude (a.u.)")
    ax.set_xlim(0, slow_time[-1])
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "target_bin_magnitude.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    res = process_fmcw()
    print("Role 2 standalone execution finished successfully.")
