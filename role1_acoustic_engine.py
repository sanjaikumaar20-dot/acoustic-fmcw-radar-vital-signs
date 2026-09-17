"""
ROLE 1: ALL-IN-ONE ACOUSTIC FMCW RADAR ENGINE & HANDOFF MODULE
==============================================================
Project: Contactless Vital Sign Tracking via Acoustic FMCW Radar

This is a self-contained, single-file drop-in module containing the complete Role 1
acoustic signal transmission, reception, simulation, diagnostics, and handoff functionality.
It requires no external file dependencies and operates 100% in software simulation.

Key Features:
1. Role1Config: Config dataclass with complete FMCW radar and target parameters.
2. Standard Simulation Engine: Generates calibrated synthetic RX audio (sample-rounded delay).
3. Continuous Fractional-Delay Physical Engine: High-fidelity sub-millimeter delay modeling.
4. Streaming / Frame-by-Frame Slicer: Generator for streaming chirps to downstream Role 2.
5. Signal Diagnostics & Forensic Metrics: In-band SNR, PSD, and matched-filter correlation.
6. Diagnostic Plotter: Generates TX chirp, RX audio, and STFT spectrogram.
7. Downstream Handoff Exporter: Exports rx_audio.npy, tx_signal.npy, and config.json.

Usage as a Standalone Script:
-----------------------------
    python role1_acoustic_engine.py

Usage as an Imported Module:
----------------------------
    from role1_acoustic_engine import Role1AcousticEngine, Role1Config

    engine = Role1AcousticEngine()
    rx_audio, tx_signal = engine.generate_signals(continuous_delay=True)
    engine.export_handoff(output_dir="path/to/role2_input")
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, Any, Generator, Optional
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


@dataclass
class Role1Config:
    """
    Configuration parameters for Acoustic FMCW Radar Simulation and Target Modeling.
    """
    # Audio Acquisition Parameters
    sample_rate_hz: int = 48000
    dtype: str = "float32"

    # FMCW Radar Waveform Parameters
    f_start_hz: float = 18000.0
    f_end_hz: float = 21000.0
    chirp_duration_s: float = 0.100
    gap_duration_s: float = 0.050
    num_chirps: int = 60

    # Target & Physical Propagation Parameters
    nominal_range_m: float = 1.00
    speed_of_sound_mps: float = 343.0
    respiration_hz: float = 0.25      # 15.0 BPM
    respiration_bpm: float = 15.0
    heartbeat_hz: float = 1.20        # 72.0 BPM
    heartbeat_bpm: float = 72.0
    respiration_displacement_mm: float = 4.0
    heartbeat_displacement_mm: float = 0.15

    # Signal & Noise Levels
    tx_amplitude: float = 0.50
    target_amplitude: float = 0.35
    direct_leakage_amplitude: float = 0.06
    noise_std: float = 0.008
    random_seed: int = 42

    # Derived Properties
    @property
    def bandwidth_hz(self) -> float:
        return self.f_end_hz - self.f_start_hz

    @property
    def chirp_rate(self) -> float:
        return self.bandwidth_hz / self.chirp_duration_s

    @property
    def carrier_frequency_hz(self) -> float:
        return self.f_start_hz + self.bandwidth_hz / 2.0

    @property
    def wavelength_m(self) -> float:
        return self.speed_of_sound_mps / self.carrier_frequency_hz

    @property
    def samples_chirp(self) -> int:
        return int(self.chirp_duration_s * self.sample_rate_hz)

    @property
    def samples_gap(self) -> int:
        return int(self.gap_duration_s * self.sample_rate_hz)

    @property
    def frame_samples(self) -> int:
        return self.samples_chirp + self.samples_gap

    @property
    def frame_duration_s(self) -> float:
        return self.chirp_duration_s + self.gap_duration_s

    @property
    def slow_time_fs(self) -> float:
        return 1.0 / self.frame_duration_s

    @property
    def theoretical_range_resolution_m(self) -> float:
        return self.speed_of_sound_mps / (2.0 * self.bandwidth_hz)

    def to_dict(self) -> Dict[str, Any]:
        """Converts config to contract JSON structure."""
        return {
            "data_type": "SIMULATED_ACOUSTIC_FMCW",
            "role1_to_role2_contract": {
                "rx_audio": "rx_audio.npy",
                "shape": [self.num_chirps * self.frame_samples],
                "dtype": self.dtype,
                "sample_rate_hz": self.sample_rate_hz
            },
            "fmcw": {
                "f_start_hz": self.f_start_hz,
                "f_end_hz": self.f_end_hz,
                "bandwidth_hz": self.bandwidth_hz,
                "chirp_duration_s": self.chirp_duration_s,
                "gap_duration_s": self.gap_duration_s,
                "num_chirps": self.num_chirps
            },
            "synthetic_target": {
                "nominal_range_m": self.nominal_range_m,
                "speed_of_sound_mps": self.speed_of_sound_mps,
                "respiration_hz": self.respiration_hz,
                "respiration_bpm": self.respiration_bpm,
                "heartbeat_hz": self.heartbeat_hz,
                "heartbeat_bpm": self.heartbeat_bpm,
                "respiration_displacement_mm": self.respiration_displacement_mm,
                "heartbeat_displacement_mm": self.heartbeat_displacement_mm
            },
            "note": "Self-contained Role 1 Acoustic FMCW simulation engine."
        }


class Role1AcousticEngine:
    """
    Complete Standalone Engine for Acoustic FMCW Radar Signal Generation & Handoff.
    """
    def __init__(self, config: Optional[Role1Config] = None):
        self.config = config if config is not None else Role1Config()
        self.rng = np.random.default_rng(self.config.random_seed)

    def generate_single_tx_chirp(self) -> np.ndarray:
        """Generates one isolated transmitted FMCW linear chirp."""
        t_chirp = np.arange(self.config.samples_chirp) / self.config.sample_rate_hz
        tx_chirp = self.config.tx_amplitude * np.cos(
            2.0 * np.pi * (
                self.config.f_start_hz * t_chirp
                + 0.5 * self.config.chirp_rate * t_chirp**2
            )
        )
        return tx_chirp.astype(self.config.dtype)

    def generate_signals(self, continuous_delay: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates the full multi-frame transmitted signal and received acoustic audio.
        
        Parameters:
        -----------
        continuous_delay : bool
            - False: Standard sample-rounded delay modeling (int(round(tau*fs))).
            - True:  Physical continuous fractional delay modeling (exact wave equation).
            
        Returns:
        --------
        rx_audio : np.ndarray
            1D float32 array containing the total acoustic microphone signal across all chirps.
        tx_signal : np.ndarray
            1D float32 array containing the transmitted chirp reference across all chirps.
        """
        cfg = self.config
        total_samples = cfg.num_chirps * cfg.frame_samples
        tx_chirp = self.generate_single_tx_chirp()

        tx_signal = np.zeros(total_samples, dtype=cfg.dtype)
        rx_audio = np.zeros(total_samples, dtype=cfg.dtype)

        t_chirp = np.arange(cfg.samples_chirp) / cfg.sample_rate_hz
        resp_m = cfg.respiration_displacement_mm / 1000.0
        card_m = cfg.heartbeat_displacement_mm / 1000.0

        for k in range(cfg.num_chirps):
            start = k * cfg.frame_samples
            end = start + cfg.samples_chirp
            tx_signal[start:end] = tx_chirp

            slow_time = k * cfg.frame_duration_s

            # Vital sign chest displacement
            displacement = (
                resp_m * np.sin(2.0 * np.pi * cfg.respiration_hz * slow_time)
                + card_m * np.sin(2.0 * np.pi * cfg.heartbeat_hz * slow_time)
            )

            target_range = cfg.nominal_range_m + displacement
            tau = 2.0 * target_range / cfg.speed_of_sound_mps

            if continuous_delay:
                # Continuous fractional delay wave propagation
                t_delayed = t_chirp - tau
                valid = t_delayed >= 0
                reflected = np.zeros(cfg.samples_chirp, dtype=np.float32)
                reflected[valid] = np.cos(
                    2.0 * np.pi * (
                        cfg.f_start_hz * t_delayed[valid]
                        + 0.5 * cfg.chirp_rate * t_delayed[valid]**2
                    )
                )
            else:
                # Sample-rounded discrete delay
                delay_samples = int(round(tau * cfg.sample_rate_hz))
                t_delayed = t_chirp - delay_samples / cfg.sample_rate_hz
                valid = t_delayed >= 0
                reflected = np.zeros(cfg.samples_chirp, dtype=np.float32)
                reflected[valid] = np.cos(
                    2.0 * np.pi * (
                        cfg.f_start_hz * t_delayed[valid]
                        + 0.5 * cfg.chirp_rate * t_delayed[valid]**2
                    )
                )

            rx_audio[start:end] += (
                cfg.direct_leakage_amplitude * tx_chirp
                + cfg.target_amplitude * reflected
            )

        # Additive acoustic background noise
        noise = self.rng.normal(0, cfg.noise_std, size=total_samples).astype(cfg.dtype)
        rx_audio += noise

        return rx_audio, tx_signal

    def stream_chirp_frames(self, continuous_delay: bool = False) -> Generator[Dict[str, Any], None, None]:
        """
        Generator for streaming one chirp frame at a time to downstream DSP processors.
        """
        rx_audio, tx_signal = self.generate_signals(continuous_delay=continuous_delay)
        cfg = self.config

        for k in range(cfg.num_chirps):
            start = k * cfg.frame_samples
            end = start + cfg.samples_chirp
            yield {
                "chirp_index": k,
                "slow_time_s": k * cfg.frame_duration_s,
                "rx_chirp": rx_audio[start:end],
                "tx_chirp": tx_signal[start:end],
                "fs_audio": cfg.sample_rate_hz,
                "config": cfg
            }

    def compute_signal_metrics(self, rx_audio: np.ndarray, tx_signal: np.ndarray) -> Dict[str, Any]:
        """Calculates RMS, peak, crest factor, in-band SNR, and matched filter correlation."""
        cfg = self.config
        rms_val = float(np.sqrt(np.mean(rx_audio**2)))
        peak_val = float(np.max(np.abs(rx_audio)))
        crest_factor_db = float(20.0 * np.log10(peak_val / (rms_val + 1e-12)))

        # Power Spectral Density
        f_psd, psd = signal.welch(rx_audio, fs=cfg.sample_rate_hz, nperseg=2048)
        in_band = (f_psd >= cfg.f_start_hz) & (f_psd <= cfg.f_end_hz)
        out_band = ~in_band & (f_psd > 1000)
        in_band_snr_db = float(10.0 * np.log10(np.sum(psd[in_band]) / (np.sum(psd[out_band]) + 1e-12)))

        # Matched filter correlation on first chirp
        chirp_tx = tx_signal[:cfg.samples_chirp]
        chirp_rx = rx_audio[:cfg.samples_chirp]
        corr = np.abs(signal.correlate(chirp_rx, chirp_tx, mode="same"))
        max_corr_norm = float(np.max(corr) / (np.sqrt(np.sum(chirp_tx**2) * np.sum(chirp_rx**2)) + 1e-12))

        return {
            "total_samples": len(rx_audio),
            "duration_seconds": len(rx_audio) / cfg.sample_rate_hz,
            "rms_amplitude": rms_val,
            "peak_amplitude": peak_val,
            "crest_factor_db": crest_factor_db,
            "in_band_snr_db": in_band_snr_db,
            "matched_filter_correlation_peak": max_corr_norm,
            "theoretical_range_resolution_mm": float(cfg.theoretical_range_resolution_m * 1000.0)
        }

    def export_handoff(
        self,
        output_dir: str = ".",
        save_continuous_benchmark: bool = True
    ) -> Dict[str, str]:
        """
        Exports all Role 1 handoff files for downstream Role 2 consumption.
        """
        os.makedirs(output_dir, exist_ok=True)
        rx_audio, tx_signal = self.generate_signals(continuous_delay=False)

        path_rx = os.path.join(output_dir, "rx_audio.npy")
        path_tx = os.path.join(output_dir, "tx_signal.npy")
        path_cfg = os.path.join(output_dir, "config.json")
        path_readme = os.path.join(output_dir, "README_ROLE2_HANDOFF.txt")

        np.save(path_rx, rx_audio)
        np.save(path_tx, tx_signal)

        with open(path_cfg, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        readme_text = (
            "ROLE 1 -> ROLE 2 HANDOFF CONTRACT\n"
            "==================================\n\n"
            "This package contains the validated simulated acoustic transmission and reception data.\n\n"
            f"MAIN ARTIFACTS:\n"
            f"  1. rx_audio.npy : 1D array of shape ({len(rx_audio)},), float32 @ {self.config.sample_rate_hz} Hz\n"
            f"  2. tx_signal.npy : 1D array of shape ({len(tx_signal)},), float32 reference FMCW chirps\n"
            f"  3. config.json  : Radar and simulation parameters\n\n"
            f"FMCW SPECIFICATIONS:\n"
            f"  • Frequency Sweep : {self.config.f_start_hz/1e3:.1f} kHz -> {self.config.f_end_hz/1e3:.1f} kHz (B = {self.config.bandwidth_hz/1e3:.1f} kHz)\n"
            f"  • Chirp Duration  : {self.config.chirp_duration_s*1e3:.0f} ms\n"
            f"  • Gap Duration    : {self.config.gap_duration_s*1e3:.0f} ms\n"
            f"  • Total Chirps    : {self.config.num_chirps} ({len(rx_audio)/self.config.sample_rate_hz:.1f} s duration)\n"
            f"  • Speed of Sound  : {self.config.speed_of_sound_mps} m/s\n\n"
            f"TARGET PARAMETERS:\n"
            f"  • Nominal Range   : {self.config.nominal_range_m:.2f} m\n"
            f"  • Respiration     : {self.config.respiration_bpm:.1f} BPM ({self.config.respiration_hz:.2f} Hz, {self.config.respiration_displacement_mm:.1f} mm)\n"
            f"  • Heartbeat       : {self.config.heartbeat_bpm:.1f} BPM ({self.config.heartbeat_hz:.2f} Hz, {self.config.heartbeat_displacement_mm:.2f} mm)\n"
        )
        with open(path_readme, "w") as f:
            f.write(readme_text)

        exported_files = {
            "rx_audio": path_rx,
            "tx_signal": path_tx,
            "config": path_cfg,
            "readme": path_readme
        }

        if save_continuous_benchmark:
            rx_cont, tx_cont = self.generate_signals(continuous_delay=True)
            path_rx_imp = os.path.join(output_dir, "rx_audio_improved.npy")
            path_tx_imp = os.path.join(output_dir, "tx_signal_improved.npy")
            np.save(path_rx_imp, rx_cont)
            np.save(path_tx_imp, tx_cont)
            exported_files["rx_audio_improved"] = path_rx_imp
            exported_files["tx_signal_improved"] = path_tx_imp

        return exported_files

    def plot_diagnostics(self, save_path: Optional[str] = None) -> None:
        """Generates comprehensive Role 1 signal verification plots."""
        rx_audio, tx_signal = self.generate_signals(continuous_delay=False)
        cfg = self.config

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig = plt.figure(figsize=(12, 8))

        # 1. Transmit Chirp Waveform
        t_chirp = np.arange(cfg.samples_chirp) / cfg.sample_rate_hz
        tx_chirp = tx_signal[:cfg.samples_chirp]
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(t_chirp * 1000, tx_chirp, color="#1f77b4", lw=1.2)
        ax1.set_title(f"Role 1 Transmitted FMCW Chirp Signal ({cfg.f_start_hz/1e3:.1f} - {cfg.f_end_hz/1e3:.1f} kHz, Tc = {cfg.chirp_duration_s*1e3:.0f} ms)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Amplitude")
        ax1.set_xlim(0, cfg.chirp_duration_s * 1000)

        # 2. Received 60-Frame Signal
        t_rx = np.arange(len(rx_audio)) / cfg.sample_rate_hz
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(t_rx, rx_audio, color="#2ca02c", lw=0.8, alpha=0.85)
        ax2.set_title(f"Role 1 Received Synthetic Audio Signal ({cfg.num_chirps} Frames, {len(rx_audio)/cfg.sample_rate_hz:.1f} s @ {cfg.sample_rate_hz} Hz)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.set_xlim(0, t_rx[-1])

        # 3. Spectrogram
        ax3 = fig.add_subplot(3, 1, 3)
        f_spec, t_spec, Sxx = signal.spectrogram(rx_audio, fs=cfg.sample_rate_hz, nperseg=1024, noverlap=512)
        band_mask = (f_spec >= 16000) & (f_spec <= 23000)
        pcm = ax3.pcolormesh(t_spec, f_spec[band_mask] / 1000, 10 * np.log10(Sxx[band_mask, :] + 1e-12),
                              shading="gouraud", cmap="magma")
        fig.colorbar(pcm, ax=ax3, label="PSD (dB/Hz)")
        ax3.set_title("Time-Frequency Spectrogram (Repeated FMCW Sweeps in Ultrasonic Band)", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Frequency (kHz)")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200)
            print(f"Diagnostics plot saved to: {save_path}")
        else:
            plt.show()
        plt.close()


def main():
    """Standalone CLI entry point."""
    print("=" * 70)
    print("    ROLE 1: ALL-IN-ONE ACOUSTIC FMCW RADAR ENGINE")
    print("=" * 70)
    
    engine = Role1AcousticEngine()
    print("FMCW Radar Configuration:")
    print(f"  • Sweep: {engine.config.f_start_hz/1e3:.1f} kHz -> {engine.config.f_end_hz/1e3:.1f} kHz (Bandwidth: {engine.config.bandwidth_hz/1e3:.1f} kHz)")
    print(f"  • Timing: Tc = {engine.config.chirp_duration_s*1e3:.0f} ms | Gap = {engine.config.gap_duration_s*1e3:.0f} ms | Frame = {engine.config.frame_duration_s*1e3:.0f} ms")
    print(f"  • Target: Nominal Range = {engine.config.nominal_range_m:.2f} m | Respiration = {engine.config.respiration_bpm:.1f} BPM | Heartbeat = {engine.config.heartbeat_bpm:.1f} BPM")
    
    rx_audio, tx_signal = engine.generate_signals(continuous_delay=False)
    metrics = engine.compute_signal_metrics(rx_audio, tx_signal)
    
    print("\nSignal Integrity & Diagnostics:")
    print(f"  • Total Duration: {metrics['duration_seconds']:.2f} s ({metrics['total_samples']} samples)")
    print(f"  • RMS Amplitude:  {metrics['rms_amplitude']:.4f} | Peak: {metrics['peak_amplitude']:.4f} (Crest Factor: {metrics['crest_factor_db']:.2f} dB)")
    print(f"  • In-Band SNR:    {metrics['in_band_snr_db']:.2f} dB (18 - 21 kHz)")
    print(f"  • Matched Filter: {metrics['matched_filter_correlation_peak']:.4f} normalized correlation peak")
    print(f"  • Range Res:      {metrics['theoretical_range_resolution_mm']:.2f} mm")

    exported = engine.export_handoff(output_dir="outputs")
    print("\nExported Downstream Handoff Files:")
    for k, p in exported.items():
        print(f"  • {k:20s}: {p}")

    engine.plot_diagnostics(save_path="plots/role1_standalone_diagnostics.png")
    print("\nRole 1 execution completed successfully.\n" + "=" * 70)


if __name__ == "__main__":
    main()
