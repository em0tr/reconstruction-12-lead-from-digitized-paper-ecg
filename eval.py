from typing import Any

import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

TOTAL_NUM_LEADS = 12
SAMPLING_RATE = 500 # Data is retrieved from records500


def plot_ecg(org_ecg: Any, rec_ecg: Any, ecg_num: int = 0, lead_num: int = 0) -> None:
    """
    Plot the original and reconstructed ECG signal.
    :param org_ecg: The original ECG signal.
    :param rec_ecg: The reconstructed ECG signal.
    :param ecg_num: The index of the ECG signal.
    :param lead_num: The lead to plot.
    :return: None
    """
    assert ecg_num < len(org_ecg), f'ecg_num must be less than {len(org_ecg)}'
    assert lead_num < TOTAL_NUM_LEADS, f'lead_num must be less than {TOTAL_NUM_LEADS}'

    plt.figure(figsize=(10, 5))
    plt.plot(org_ecg[ecg_num, :, lead_num], label='original')
    plt.plot(rec_ecg[ecg_num, :, lead_num], label='reconstructed', alpha=0.7)
    plt.title(f'ECG {ecg_num} – Lead {lead_num}')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()


def load_data(data: Any) -> tuple[Any, Any]:
    """
    Load the data.
    :param data: The npy.npz file after running main.ipynb
    :return: A tuple of the original and reconstructed ECG signal.
    """
    with np.load(data) as data:
        org = data['original']
        rec = data['reconstructed']
    return org, rec




if __name__ == '__main__':
    original, reconstructed = load_data('input/data.npy.npz')

    index = 0  # Which ECG to plot
    lead = 0  # Which lead to plot
    plot_ecg(original, reconstructed, ecg_num=index, lead_num=lead)