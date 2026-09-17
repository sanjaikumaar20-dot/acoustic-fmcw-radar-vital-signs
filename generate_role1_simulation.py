import numpy as np

fs = 48000
f_start = 18000
f_end = 21000
B = f_end - f_start
chirp_duration = 0.10
gap_duration = 0.05
num_chirps = 60

c = 343.0
target_range_m = 1.0

respiration_hz = 0.25
heartbeat_hz = 1.20
respiration_displacement_m = 0.004
heartbeat_displacement_m = 0.00015

target_amplitude = 0.35
direct_leakage_amplitude = 0.06
noise_std = 0.008

rng = np.random.default_rng(42)

samples_chirp = int(chirp_duration * fs)
samples_gap = int(gap_duration * fs)
t_chirp = np.arange(samples_chirp) / fs

tx_chirp = np.cos(
    2 * np.pi * (
        f_start * t_chirp
        + 0.5 * (B / chirp_duration) * t_chirp**2
    )
)
tx_chirp *= 0.5

frame_len = samples_chirp + samples_gap
total_samples = num_chirps * frame_len

tx_signal = np.zeros(total_samples)
rx_audio = np.zeros(total_samples)

for k in range(num_chirps):
    start = k * frame_len
    end = start + samples_chirp
    tx_signal[start:end] = tx_chirp

    slow_time = k * frame_len / fs

    displacement = (
        respiration_displacement_m *
        np.sin(2 * np.pi * respiration_hz * slow_time)
        +
        heartbeat_displacement_m *
        np.sin(2 * np.pi * heartbeat_hz * slow_time)
    )

    target_range = target_range_m + displacement
    tau = 2 * target_range / c
    delay_samples = int(round(tau * fs))

    t_delayed = t_chirp - delay_samples / fs
    valid = t_delayed >= 0
    reflected = np.zeros(samples_chirp)

    reflected[valid] = np.cos(
        2 * np.pi * (
            f_start * t_delayed[valid]
            + 0.5 * (B / chirp_duration) * t_delayed[valid]**2
        )
    )

    rx_audio[start:end] += (
        direct_leakage_amplitude * tx_chirp
        + target_amplitude * reflected
    )

rx_audio += rng.normal(0, noise_std, size=total_samples)

np.save("rx_audio.npy", rx_audio.astype(np.float32))
np.save("tx_signal.npy", tx_signal.astype(np.float32))

print("Created rx_audio.npy")
print("Created tx_signal.npy")
print("Shape:", rx_audio.shape)
print("Sample rate:", fs, "Hz")
print("Synthetic target:", target_range_m, "m")
print("Synthetic respiration:", respiration_hz * 60, "BPM")
print("Synthetic heart rate:", heartbeat_hz * 60, "BPM")
