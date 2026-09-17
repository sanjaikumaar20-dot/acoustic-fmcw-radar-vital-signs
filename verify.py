import numpy as np

rx_audio = np.load("rx_audio.npy")

print("Shape:", rx_audio.shape)
print("Data type:", rx_audio.dtype)
print("Duration:", len(rx_audio) / 48000, "seconds")
print("Minimum:", rx_audio.min())
print("Maximum:", rx_audio.max())
