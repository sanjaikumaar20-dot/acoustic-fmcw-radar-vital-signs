ROLE 1 -> ROLE 2 HANDOFF
==========================

This package is a SIMULATED replacement for the real microphone acquisition.

MAIN CONTRACT:
    rx_audio.npy

It contains:
    - 1D NumPy array
    - float32
    - sampled at 48 kHz

Synthetic FMCW:
    18 kHz -> 21 kHz
    chirp duration = 100 ms
    gap = 50 ms
    60 chirps
    total duration = 9 seconds

Synthetic target:
    nominal range = 1.0 m
    respiration = 15 BPM
    heartbeat = 72 BPM

IMPORTANT:
This is synthetic validation data, NOT a real human measurement.

FOR ROLE 2:
1. Copy rx_audio.npy into the Role 2 project folder.
2. Load it with:
       rx_audio = np.load("rx_audio.npy")
3. Use fs = 48000.
4. Use the known transmitted chirp in tx_signal.npy as the reference,
   or regenerate the same 18-21 kHz chirp.
5. Perform:
       segment chirps
       -> Hilbert/analytic signal if required
       -> dechirp with conjugate TX
       -> fast-time FFT
       -> range peak detection
       -> slow-time target-bin extraction

Expected synthetic target:
    around 1.0 m range
    with slow-time vital-sign modulation.

The simulated data is deliberately clean enough to make the
Role-2 -> Role-3 handoff possible quickly.
