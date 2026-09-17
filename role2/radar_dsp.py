"""
Role 2: Modular FMCW Radar DSP Architecture
===========================================
Modules:
- FMCWRangeConfig: Radar configuration and physical conversion parameters.
- DechirpEngine: Analytic signal mixing and beat generation.
- RangeFFTEngine: Windowed fast-time FFT with physical distance mapping.
- StaticClutterCanceller: Slow-time exponential moving average (EMA) clutter canceller.
- TargetBinSelector: Dynamic peak detection and target bin identification.
"""

from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np
import scipy.signal as signal


@dataclass
class FMCWRangeConfig:
    fs_audio: int = 48000
    f_start: float = 18000.0
    f_end: float = 21000.0
    chirp_duration_sec: float = 0.100
    gap_duration_sec: float = 0.050
    speed_of_sound: float = 343.0
    n_fft: int = 19200
    min_target_range_m: float = 0.50
    max_target_range_m: float = 2.00
    clutter_alpha: float = 0.90

    @property
    def bandwidth(self) -> float:
        return self.f_end - self.f_start

    @property
    def chirp_rate(self) -> float:
        return self.bandwidth / self.chirp_duration_sec

    @property
    def carrier_frequency(self) -> float:
        return self.f_start + self.bandwidth / 2.0

    @property
    def wavelength(self) -> float:
        return self.speed_of_sound / self.carrier_frequency

    @property
    def samples_chirp(self) -> int:
        return int(self.chirp_duration_sec * self.fs_audio)

    @property
    def samples_gap(self) -> int:
        return int(self.gap_duration_sec * self.fs_audio)

    @property
    def frame_samples(self) -> int:
        return self.samples_chirp + self.samples_gap

    def beat_freq_to_range(self, f_beat: np.ndarray) -> np.ndarray:
        return self.speed_of_sound * f_beat * self.chirp_duration_sec / (2.0 * self.bandwidth)

    def range_to_beat_freq(self, r_m: float) -> float:
        return 2.0 * self.bandwidth * r_m / (self.speed_of_sound * self.chirp_duration_sec)


class DechirpEngine:
    """Performs analytic Hilbert mixing between transmit and receive chirps."""
    def __init__(self, config: FMCWRangeConfig):
        self.config = config
        t_chirp = np.arange(config.samples_chirp) / config.fs_audio
        tx_ref = np.cos(2 * np.pi * (config.f_start * t_chirp + 0.5 * config.chirp_rate * t_chirp**2))
        self.tx_analytic = signal.hilbert(tx_ref)

    def process_segment(self, rx_segment: np.ndarray) -> np.ndarray:
        rx_analytic = signal.hilbert(rx_segment)
        # s_beat = tx * conj(rx)
        return self.tx_analytic * np.conj(rx_analytic)


class RangeFFTEngine:
    """Computes fast-time windowed FFT and calibrated range axis."""
    def __init__(self, config: FMCWRangeConfig):
        self.config = config
        self.window = np.hamming(config.samples_chirp)
        self.freq_axis = np.fft.fftfreq(config.n_fft, d=1/config.fs_audio)[:config.n_fft // 2]
        self.range_axis = config.beat_freq_to_range(self.freq_axis)

    def compute_range_profile(self, beat_signal: np.ndarray) -> np.ndarray:
        spec = np.fft.fft(beat_signal * self.window, n=self.config.n_fft)
        return spec[:self.config.n_fft // 2]


class StaticClutterCanceller:
    """Slow-time exponential moving average (EMA) clutter cancellation filter."""
    def __init__(self, config: FMCWRangeConfig):
        self.alpha = config.clutter_alpha
        self.ema_profile = None

    def filter(self, range_profile: np.ndarray) -> np.ndarray:
        if self.ema_profile is None:
            self.ema_profile = range_profile.copy()
            return range_profile - self.ema_profile
        self.ema_profile = self.alpha * self.ema_profile + (1.0 - self.alpha) * range_profile
        return range_profile - self.ema_profile
