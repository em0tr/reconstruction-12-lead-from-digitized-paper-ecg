import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

TOTAL_NUM_LEADS = 12
SAMPLING_RATE = 500 # Data is retrieved from records500
DATA_SHAPE = (4368, 5000, 12) # Shape of the original and reconstructed data


def plot_ecg(org_ecg, rec_ecg, ecg_num: int = 0, lead_num: int = 0) -> None:
    """
    Plot the original and reconstructed ECG signal using matplotlib.
    :param org_ecg: The original ECG signal.
    :param rec_ecg: The reconstructed ECG signal.
    :param ecg_num: The index of the ECG signal.
    :param lead_num: The lead to plot.
    :return: None
    """
    check_boundaries(org_ecg, ecg_num, lead_num)
    check_boundaries(rec_ecg, ecg_num, lead_num)

    plt.figure(figsize=(10, 5))
    plt.plot(org_ecg[ecg_num, :, lead_num], label='original')
    plt.plot(rec_ecg[ecg_num, :, lead_num], label='reconstructed', alpha=0.7)
    plt.title(f'ECG {ecg_num} – Lead {lead_num}')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()


def load_data(data):
    """
    Load the data.
    :param data: The npy.npz file after running main.ipynb
    :return: A tuple of the original and reconstructed ECG signal.
    """
    if not os.path.exists(data):
        raise FileNotFoundError(f'File {data} not found.')
    with np.load(data) as data:
        org = data['original']
        rec = data['reconstructed']
    return org, rec


def check_boundaries(data, ecg_num, lead_num) -> None:
    """
    Check if the ECG signal has the required boundaries. For example there is only 12 leads
    so trying to use lead 13 which doesn't exist is asserted.
    :param data: The original or reconstructed ECG signal.
    :param ecg_num: The index of the ECG signal.
    :param lead_num: The lead.
    :return: None
    """
    assert ecg_num < len(data), f'ecg_num must be less than {len(data)}'
    assert lead_num < TOTAL_NUM_LEADS, f'lead_num must be less than {TOTAL_NUM_LEADS}'


def plot_nk_r_peaks(data, ecg_num: int = 0, lead_num: int = 0, zoom: bool = False, zoom_level: int = 5) -> None:
    """
    Plot the R peaks of original or reconstructed ECG signal using neurokit2.
    :param data: The ECG signal.
    :param ecg_num: The index of the ECG signal.
    :param lead_num: The lead to plot.
    :param zoom: Whether to zoom the plot.
    :param zoom_level: The zoom level. A zoom level of 5 will only show 5 peaks.
    :return: None
    """
    check_boundaries(data, ecg_num, lead_num)
    if data.shape == DATA_SHAPE:
        ecg_signal = get_ecg(data, ecg_num, lead_num)
    else:
        ecg_signal = data
    _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=SAMPLING_RATE)
    if zoom:
        plot = nk.events_plot(rpeaks['ECG_R_Peaks'][:zoom_level], ecg_signal[:zoom_level * SAMPLING_RATE])
    else:
        plot = nk.events_plot(rpeaks['ECG_R_Peaks'], ecg_signal)
    plt.grid()
    plt.title(f'$R$ peaks of ECG {ecg_num} – Lead {lead_num}')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()


def get_ecg(data, ecg_num, lead_num):
    """
    Get single ECG signal.
    :param data: The original or reconstructed ECG signal.
    :param ecg_num: The index of the ECG signal.
    :param lead_num: The lead to plot.
    :return: ECG signal.

    Example: To get the first ECG with the first lead of the original data:
    ::
        ecg_signal = get_ecg(original, 0, 0)
    """
    return data[ecg_num, :, lead_num]

if __name__ == '__main__':
    original, reconstructed = load_data('output/data.npy.npz')
    plot_nk_r_peaks(original, 0, 0)