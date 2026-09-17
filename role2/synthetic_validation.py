"""
Role 2: Synthetic FMCW Radar Validation Suite
=============================================
Synthesizes FMCW return signals with targets at 0.5 m, 1.0 m, and 1.5 m
to validate range profiling accuracy, range resolution, and multi-target detection.
"""

import os
import json
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from role2.radar_dsp import FMCWRangeConfig, DechirpEngine, RangeFFTEngine


def run_synthetic_validation(config_path="config.json", output_dir="outputs", plot_dir="plots"):
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    if not os.path.isabs(config_path):
        config_path = os.path.join(base_dir, "..", config_path)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(base_dir, "..", output_dir)
    if not os.path.isabs(plot_dir):
        plot_dir = os.path.join(base_dir, "..", plot_dir)

    config = FMCWRangeConfig()
    dechirp = DechirpEngine(config)
    fft_engine = RangeFFTEngine(config)

    test_targets = [0.50, 1.00, 1.50]  # Target distances in meters
    target_amps = [0.3, 0.4, 0.25]
    
    t = np.arange(config.samples_chirp) / config.fs_audio
    K = config.chirp_rate
    
    # Generate multi-target synthetic RX signal
    rx_synth = np.zeros(config.samples_chirp)
    for dist, amp in zip(test_targets, target_amps):
        tau = 2.0 * dist / config.speed_of_sound
        t_delayed = t - tau
        valid = t_delayed >= 0
        rx_synth[valid] += amp * np.cos(2 * np.pi * (config.f_start * t_delayed[valid] + 0.5 * K * t_delayed[valid]**2))

    # Add Gaussian noise
    rx_synth += np.random.default_rng(42).normal(0, 0.005, size=config.samples_chirp)

    # Process through Dechirp and Fast-time FFT
    beat = dechirp.process_segment(rx_synth)
    range_profile = fft_engine.compute_range_profile(beat)
    mag_profile = np.abs(range_profile)

    # Find peaks
    peaks, _ = signal.find_peaks(mag_profile, height=np.max(mag_profile) * 0.20, distance=10)
    detected_ranges = [float(fft_engine.range_axis[p]) for p in peaks if fft_engine.range_axis[p] <= 2.2]

    return {
        "true_targets_m": test_targets,
        "detected_targets_m": detected_ranges,
        "range_resolution_m": float(config.speed_of_sound / (2 * config.bandwidth))
    }
