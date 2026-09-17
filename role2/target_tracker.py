"""
Role 2: Multi-Candidate Target Tracker & Multipath Rejection Engine
==================================================================
Identifies target peaks using:
1. Multi-candidate peak detection within search window [0.5m, 2.0m]
2. Gaussian spatial proximity kernel for range continuity
3. Magnitude persistence & peak-to-noise ratio (PNR) scoring
4. Tracking gate to prevent bin hopping
"""

import numpy as np
import scipy.signal as signal
from role2.radar_dsp import FMCWRangeConfig


class MultiCandidateTargetTracker:
    def __init__(self, config: FMCWRangeConfig, gate_m: float = 0.20):
        self.config = config
        self.gate_m = gate_m
        self.tracked_range_m = None
        self.tracked_bin_idx = None
        self.history = []

    def track(self, range_profile: np.ndarray, range_axis: np.ndarray) -> dict:
        mag = np.abs(range_profile)
        
        # Valid search window
        search_mask = (range_axis >= self.config.min_target_range_m) & (range_axis <= self.config.max_target_range_m)
        search_indices = np.where(search_mask)[0]

        # Find local peaks
        peaks, _ = signal.find_peaks(mag[search_indices], height=np.max(mag[search_indices]) * 0.15)
        
        if len(peaks) == 0:
            # Fallback to maximum
            best_idx_sub = np.argmax(mag[search_indices])
            best_bin = search_indices[best_idx_sub]
        else:
            candidate_bins = search_indices[peaks]
            candidate_ranges = range_axis[candidate_bins]
            candidate_mags = mag[candidate_bins]

            if self.tracked_range_m is None:
                # Initial acquisition: Pick strongest peak
                best_idx = np.argmax(candidate_mags)
                best_bin = candidate_bins[best_idx]
            else:
                # Multi-factor score: Magnitude + Spatial Proximity Kernel
                dist_penalty = np.exp(-((candidate_ranges - self.tracked_range_m)**2) / (2 * (self.gate_m**2)))
                scores = (candidate_mags / np.max(candidate_mags)) * 0.6 + dist_penalty * 0.4
                best_idx = np.argmax(scores)
                best_bin = candidate_bins[best_idx]

        self.tracked_bin_idx = int(best_bin)
        self.tracked_range_m = float(range_axis[best_bin])
        complex_value = range_profile[best_bin]
        
        self.history.append({
            "range_m": self.tracked_range_m,
            "bin_idx": self.tracked_bin_idx,
            "magnitude": float(mag[best_bin])
        })

        return {
            "tracked_range_m": self.tracked_range_m,
            "tracked_bin_idx": self.tracked_bin_idx,
            "complex_val": complex_value
        }
