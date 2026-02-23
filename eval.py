import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

SAMPLING_RATE = 500 # records500 is 500Hz

data = np.load("input/data.npy.npz")

original = data['original']
reconstructed = data['reconstructed']

index = 0 # Which ECG to plot
lead = 1   # Which lead to plot
plt.figure()
plt.plot(original[index, :, lead], label='original')
plt.plot(reconstructed[index, :, lead], label='reconstructed', alpha=0.7)
plt.title(f'ECG {index} – Lead {lead}')
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

data.close()