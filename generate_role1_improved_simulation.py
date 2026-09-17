"""
IMPROVED ROLE 1: Physically Consistent Acoustic FMCW Simulation
===============================================================
This module creates a physically consistent synthetic FMCW dataset using
continuous fractional-delay wave propagation rather than integer sample-rounding.

Why this is physically necessary:
--------------------------------
In the original generate_role1_simulation.py:
    delay_samples = int(round(tau * fs))
    t_delayed = t_chirp - delay_samples / fs

At fs = 48000 Hz and c = 343 m/s:
    1 sample delay = c / (2 * fs) = 3.57 mm of round-trip displacement.
    Respiration (4.0 mm) causes coarse 1-sample staircase steps.
    Cardiac motion (0.15 mm = 0.042 samples) is completely buried below 1 sample!

In this improved generator:
    tau(t) = 2 * (R_nominal + d_resp(t) + d_card(t)) / c
    t_delayed = t_chirp - tau(t)  [Exact continuous fractional delay]
    reflected(t) = cos(2*pi * (f_start * t_delayed + 0.5 * K * t_delayed^2))

This preserves continuous carrier phase modulation down to micrometers,
allowing exact recovery of both respiration (15 BPM) and heartbeat (72 BPM).
"""

import os
import json
import numpy as np


def generate_improved_simulation(
    config_path="config.json",
    out_rx="rx_audio_improved.npy",
    out_tx="tx_signal_improved.npy",
    num_chirps_override=None
):
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    
    if not os.path.isabs(config_path):
        config_path = os.path.join(base_dir, config_path)
    if not os.path.isabs(out_rx):
        out_rx = os.path.join(base_dir, out_rx)
    if not os.path.isabs(out_tx):
        out_tx = os.path.join(base_dir, out_tx)

    with open(config_path, "r") as f:
        config = json.load(f)

    fs = int(config["role1_to_role2_contract"]["sample_rate_hz"])
    f_start = float(config["fmcw"]["f_start_hz"])
    f_end = float(config["fmcw"]["f_end_hz"])
    B = float(config["fmcw"]["bandwidth_hz"])
    Tc = float(config["fmcw"]["chirp_duration_s"])
    Tgap = float(config["fmcw"]["gap_duration_s"])
    num_chirps = num_chirps_override if num_chirps_override else int(config["fmcw"]["num_chirps"])
    c = float(config["synthetic_target"]["speed_of_sound_mps"])
    target_range_m = float(config["synthetic_target"]["nominal_range_m"])

    respiration_hz = float(config["synthetic_target"]["respiration_hz"])
    heartbeat_hz = float(config["synthetic_target"]["heartbeat_hz"])
    respiration_displacement_m = float(config["synthetic_target"]["respiration_displacement_mm"]) / 1000.0
    heartbeat_displacement_m = float(config["synthetic_target"]["heartbeat_displacement_mm"]) / 1000.0

    target_amplitude = 0.35
    direct_leakage_amplitude = 0.06
    noise_std = 0.008

    rng = np.random.default_rng(42)

    samples_chirp = int(Tc * fs)
    samples_gap = int(Tgap * fs)
    frame_len = samples_chirp + samples_gap
    total_samples = num_chirps * frame_len

    t_chirp = np.arange(samples_chirp) / fs
    K = B / Tc

    tx_chirp = 0.5 * np.cos(2 * np.pi * (f_start * t_chirp + 0.5 * K * t_chirp**2))
    tx_signal = np.zeros(total_samples, dtype=np.float32)
    rx_audio = np.zeros(total_samples, dtype=np.float32)

    for k in range(num_chirps):
        start = k * frame_len
        end = start + samples_chirp
        tx_signal[start:end] = tx_chirp

        slow_time = k * frame_len / fs

        displacement = (
            respiration_displacement_m * np.sin(2 * np.pi * respiration_hz * slow_time)
            + heartbeat_displacement_m * np.sin(2 * np.pi * heartbeat_hz * slow_time)
        )

        target_range = target_range_m + displacement
        # Exact continuous fractional delay
        tau = 2.0 * target_range / c
        
        t_delayed = t_chirp - tau
        valid = t_delayed >= 0
        reflected = np.zeros(samples_chirp)

        reflected[valid] = np.cos(
            2 * np.pi * (f_start * t_delayed[valid] + 0.5 * K * t_delayed[valid]**2)
        )

        rx_audio[start:end] += (
            direct_leakage_amplitude * tx_chirp
            + target_amplitude * reflected
        )

    rx_audio += rng.normal(0, noise_std, size=total_samples).astype(np.float32)

    np.save(out_rx, rx_audio)
    np.save(out_tx, tx_signal)
    print(f"Created physically consistent dataset: {out_rx} ({num_chirps} chirps)")
    return out_rx, out_tx


if __name__ == "__main__":
    generate_improved_simulation()
