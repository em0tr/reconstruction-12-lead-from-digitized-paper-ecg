import numpy as np
import neurokit2 as nk
import os
import matplotlib.pyplot as plt
from enum import Enum

SAMPLING_RATE = 500  # Data is retrieved from records500
LEAD_LABELS = [
    'I', 'II', 'III',
    'aVR', 'aVL', 'aVF',
    'V1', 'V2', 'V3',
    'V4', 'V5', 'V6',
]
TOTAL_NUM_LEADS = len(LEAD_LABELS)
TOTAL_NUM_ECGS = 0#15#4368


class TypeECG(Enum):
    ORIGINAL = 1
    RECONSTRUCTED = 2


class ECG:
    def __init__(self, input_data, type_ecg: TypeECG):
        self.type_ecg = type_ecg
        self.data = self.load_data(input_data)
        self.ecg_r_peaks = {}
        self.ecg_t_peaks = {}
        self.ecg_p_peaks = {}
        self.ecg_q_peaks = {}
        self.ecg_s_peaks = {}

    def get_r_peaks_empty(self):
        """
        Check if the ecg_r_peaks dictionary is empty since certain operations requires it to be not empty.
        :return: True if empty, False otherwise.
        """
        if not self.ecg_r_peaks:
            return True
        return False

    def get_r_peaks(self, ecg_number: int, lead_number: int, r_value: int=None):
        """
        Get the R-peaks for an ECG and lead signal.
        :param ecg_number: The ECG number.
        :param lead_number: The lead number.
        :param r_value: Used to get a specific R-peak value from an ECG and lead signal.
        :return: The R-peaks for an ECG and lead signal.
        """
        self.check_boundaries(ecg_number, lead_number)
        if r_value is not None:
            return self.ecg_r_peaks[(ecg_number, slice(None, None, None), lead_number)][r_value]
        # TODO: Maybe redefine the keys for the R-peaks to drop the slice() bit
        return self.ecg_r_peaks[(ecg_number, slice(None, None, None), lead_number)]

    def get_t_peaks(self, ecg_number: int, lead_number: int, t_value: int=None):
        self.check_boundaries(ecg_number, lead_number)
        if t_value is not None:
            return self.ecg_t_peaks[ecg_number, :, lead_number][t_value]
        return self.ecg_t_peaks[ecg_number, :, lead_number]

    def get_p_peaks(self, ecg_number: int, lead_number: int=None, p_value: int=None):
        self.check_boundaries(ecg_number, lead_number)
        if p_value is not None:
            return self.ecg_p_peaks[ecg_number, :, lead_number][p_value]
        return self.ecg_p_peaks[ecg_number, :, lead_number]

    def get_q_peaks(self, ecg_number: int, lead_number: int, q_value: int=None):
        self.check_boundaries(ecg_number, lead_number)
        if q_value is not None:
            return self.ecg_q_peaks[ecg_number, :, lead_number][q_value]
        return self.ecg_q_peaks[ecg_number, :, lead_number]

    def get_s_peaks(self, ecg_number: int, lead_number: int, s_value: int=None):
        self.check_boundaries(ecg_number, lead_number)
        if s_value is not None:
            return self.ecg_s_peaks[ecg_number, :, lead_number][s_value]
        return self.ecg_s_peaks[ecg_number, :, lead_number]

    def get_shape(self):
        return self.data.shape

    def __eq__(self, other):
        """
        Compare the shape of the data as both the original and reconstructed data should have the same shape.
        :param other: The other ECG object.
        :return:
        """
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.get_shape() == other.get_shape()

    def load_data(self, data):
        """
        Load the data in the data.npy.npz file.
        :param data: The npy.npz file after running main.ipynb
        :return: The original or reconstructed data.
        """
        if not os.path.exists(data):
            raise FileNotFoundError(f'File {data} not found.')
        with np.load(data) as data:
            if self.type_ecg == TypeECG.ORIGINAL:
                v = data['original']
            else:
                v = data['reconstructed']
        return v

    def find_r_peaks(self, df, ecg_start=0, ecg_end=TOTAL_NUM_ECGS, lead_start=0, lead_end=TOTAL_NUM_LEADS):
        """
        Find R-peaks for a range of ECGs and leads and store the peaks and mean of peaks.
        :param df: The dataframe to save the r_peak mean values to.
        :param ecg_start: The start ECG index.
        :param ecg_end: The end ECG index.
        :param lead_start: The lead start index.
        :param lead_end: The lead end index.
        :return: None
        """
        print(f'Calculating mean of R-peaks on {self.get_ecg_type()} ECG(s) [{ecg_start} - {ecg_end - 1}] using '
              f'lead(s) [{LEAD_LABELS[lead_start]} - {LEAD_LABELS[lead_end - 1]}]')
        for ecg in range(ecg_start, ecg_end):
            for lead in range(lead_start, lead_end):
                _, r_peaks_top = self.get_r_peaks_from_signal(ecg_num=ecg, lead_num=lead)
                r_peak_top_mean = self.calculate_peak_mean(r_peaks_top, ecg, lead, print_r_peaks=False, print_mean_values=False)
                df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 'r_peak_mean'] = r_peak_top_mean

    def calculate_peak_mean(self, peaks, ecg_num, lead_num, print_mean_values=False, print_r_peaks=False):
        """
        Calculate the R-peak mean.
        :param peaks: The R-peaks.
        :param ecg_num: The index of the ECG signal.
        :param lead_num: The lead.
        :param print_mean_values: Print the mean of the R-peaks of the original and reconstructed signals.
        :param print_r_peaks: Print the R-peaks of the original and reconstructed signals.
        :return: The mean of the original and reconstructed signals.
        """
        mean = np.mean(peaks)
        if print_r_peaks:
            self.print_peaks(peaks)
        if print_mean_values:
            print(f'Mean value of original R-peaks for ECG {ecg_num} and lead {LEAD_LABELS[lead_num]}: {mean:.4f}')
        return mean

    def print_peaks(self, data):
        """
        Print the R-peaks from the ECG signal to the terminal.
        :param data: The ECG signal with R-peaks.
        :return: None
        """
        if self.type_ecg == TypeECG.ORIGINAL:
            print(f'Original peaks: Total {len(data)} peaks\n{data}')
        else:
            print(f'Reconstructed peaks: Total {len(data)} peaks\n{data}')

    def get_r_peaks_from_signal(self, ecg_num, lead_num):
        """
        Get the R-peaks from an ECG and lead signal.
        :param ecg_num: The ECG index.
        :param lead_num: The lead
        :return: The R-peaks from the signal.
        """
        ecg, signal, r_peaks, r_peak_tops = self.process_signal(ecg_num, lead_num)
        self.ecg_r_peaks[ecg_num, :, lead_num] = r_peaks["ECG_R_Peaks"]
        if self.type_ecg == TypeECG.ORIGINAL:
            title = f'- Original - ECG {ecg_num} - Lead {LEAD_LABELS[lead_num]}'
        else:
            title = f'- Reconstructed - ECG {ecg_num} - Lead {LEAD_LABELS[lead_num]}'
        peaks = self.plot_r_peaks(ecg, r_peaks,
                                  title_suffix=title,
                                  use_plotting=False,
                                  zoom=False,
                                  zoom_level=3)
        return peaks, r_peak_tops

    @staticmethod
    def plot_r_peaks(ecg_signal, info, title_suffix='', zoom=False, zoom_level=5, use_plotting=False):
        """
        After getting the R-peaks, the R-peaks can be plotted.
        :param ecg_signal: The ECG signal.
        :param info: The ECG information.
        :param title_suffix: The plot title.
        :param zoom: Zoom in on plots to focus only on a few peaks.
        :param zoom_level: The zoom level.
        :param use_plotting: Plot the R-peaks.
        :return: R-peaks.
        """
        rpeaks = info['ECG_R_Peaks']
        if use_plotting:
            if zoom:
                max_idx = int(zoom_level * SAMPLING_RATE)
                mask = (rpeaks < max_idx)
                rpeaks_to_plot = rpeaks[mask]
                ecg_to_plot = ecg_signal[:max_idx]
            else:
                rpeaks_to_plot = rpeaks
                ecg_to_plot = ecg_signal
            nk.events_plot(rpeaks_to_plot, ecg_to_plot)
            plt.grid(True, alpha=0.3)
            plt.title(f'$R$-peaks {title_suffix}')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            plt.tight_layout()
            plt.show()
        return rpeaks

    def process_signal(self, ecg_num, lead_num):
        """
        Process the ECG signal using NeuroKit's preprocessing pipeline.
        :param ecg_num: The index of the ECG signal.
        :param lead_num: The lead.
        :return: A processed ECG signal, its signal information and the R-peaks.
        """
        ecg = self.get_ecg(ecg_num, lead_num)
        ecg, _ = nk.ecg_invert(ecg, sampling_rate=SAMPLING_RATE)
        signals, r_peaks = nk.ecg_process(ecg, sampling_rate=SAMPLING_RATE, method='neurokit')
        r_peak_tops = ecg[r_peaks["ECG_R_Peaks"]]
        return ecg, signals, r_peaks, r_peak_tops

    def get_ecg(self, ecg_num: int, lead_num: int):
        """
        Get and clean a single ECG signal.
        :param ecg_num: The index of the ECG signal.
        :param lead_num: The lead to plot.
        :return: ECG signal.

        Example: To get the first ECG with the first lead of the original data:
        ::
            ecg_signal = original.get_ecg(0, 0)
        """
        self.check_boundaries(ecg_num, lead_num)
        ecg = self.data[ecg_num, :, lead_num]
        cleaned_ecg = nk.ecg_clean(ecg, sampling_rate=SAMPLING_RATE)
        return cleaned_ecg

    def check_boundaries(self, ecg_num, lead_num) -> None:
        """
        Check if the ECG signal has the required boundaries. For example there is only 12 leads
        so trying to use lead 13 which doesn't exist raises an error.
        :param ecg_num: The index of the ECG signal.
        :param lead_num: The lead.
        :return: None
        """
        n, t, c = self.get_shape()  # (N, T, C) -> (4368, 5000, 12)
        if not 0 <= ecg_num < n:
            raise IndexError(f'ECG signal index {ecg_num} out of range [0, {n - 1}]')
        if not 0 <= lead_num < c:
            raise IndexError(f'Lead number {lead_num} out of range [0, {c - 1}]')

    def get_ecg_type(self):
        """
        Print the ECG type where only the first letter is capitalized.4368
        :return:
        """
        return self.type_ecg.name[0] + self.type_ecg.name[1:].lower()

    def find_ecg_peaks(self, df, ecg_start=0, ecg_end=TOTAL_NUM_ECGS,
                       lead_start=0, lead_end=TOTAL_NUM_LEADS,
                       use_plotting=False, zoom_level=4,
                       print_peaks=False, use_show=False):
        """
        Find the peaks of the ECG signals.
        :param ecg_start: The start index of the ECG signal.
        :param ecg_end:  The end index of the ECG signal.
        :param lead_start: The start index of the lead.
        :param lead_end: The end index of the lead.
        :param use_plotting: Plot the ECG peaks.
        :param zoom_level: The zoom level for the plotted ECG peaks.
        :param print_peaks: Print the different ECG peaks.
        :return: None
        """
        if self.get_r_peaks_empty():
            print('Finding the ECG peaks requires having the R-peaks, find the R-peaks before the ECG peaks.')
            raise SystemExit
        for ecg in range(ecg_start, ecg_end):
            for lead in range(lead_start, lead_end):
                current_ecg = self.get_ecg(ecg_num=ecg, lead_num=lead)
                r_peaks = self.get_r_peaks(ecg, lead)
                if use_show:
                    _, waves_peak = nk.ecg_delineate(current_ecg, r_peaks, sampling_rate=SAMPLING_RATE,
                                                     method='peak', show=True, show_type='peaks')
                else:
                    _, waves_peak = nk.ecg_delineate(current_ecg, r_peaks, sampling_rate=SAMPLING_RATE,
                                                     method='peak', show=False, show_type='peaks')
                if use_plotting:
                    plot = nk.events_plot([waves_peak['ECG_T_Peaks'][:zoom_level],
                                           waves_peak['ECG_P_Peaks'][:zoom_level],
                                           waves_peak['ECG_Q_Peaks'][:zoom_level],
                                           waves_peak['ECG_S_Peaks'][:zoom_level]],
                                          current_ecg[:(zoom_level + 1) * SAMPLING_RATE])
                    plt.grid(True, alpha=0.3)
                    plt.title(f'Peaks - {self.get_ecg_type()} ECG - {ecg} - Lead {LEAD_LABELS[lead]}')
                    plt.xlabel('Samples')
                    plt.ylabel('Amplitude')
                    plt.tight_layout()
                    plt.show()
                if print_peaks:
                    print(f'T-peaks for {self.get_ecg_type()} ECG {ecg} and '
                          f'lead {LEAD_LABELS[lead]}: {waves_peak["ECG_T_Peaks"]}')
                    print(f'P-peaks for {self.get_ecg_type()} ECG {ecg} and '
                          f'lead {LEAD_LABELS[lead]}: {waves_peak["ECG_P_Peaks"]}')
                    print(f'Q-peaks for {self.get_ecg_type()} ECG {ecg} and '
                          f'lead {LEAD_LABELS[lead]}: {waves_peak["ECG_Q_Peaks"]}')
                    print(f'S-peaks for {self.get_ecg_type()} ECG {ecg} and '
                          f'lead {LEAD_LABELS[lead]}: {waves_peak["ECG_S_Peaks"]}')
                try:
                    self.ecg_t_peaks[ecg, :, lead] = waves_peak["ECG_T_Peaks"]
                    t_peak_tops = current_ecg[self.ecg_t_peaks[ecg, :, lead]]
                    t_peak_top_mean = self.calculate_peak_mean(t_peak_tops, ecg, lead)
                    print(f'T-peaks on the samples axis for {self.get_ecg_type()} ECG {ecg} and lead {lead}: {self.ecg_t_peaks[ecg, :, lead]}')
                    print(f'T-peaks on the Amplitude for {self.get_ecg_type()} ECG {ecg} and lead {lead}: {t_peak_tops}')
                    print(f'T-peak mean for ECG {ecg} and lead {lead}: {t_peak_top_mean}:')
                    df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 't_mean'] = t_peak_top_mean
                except IndexError:
                    print(f'Error finding T-peaks for {self.get_ecg_type()} ECG {ecg} and lead {lead}')
                self.ecg_p_peaks[ecg, :, lead] = waves_peak["ECG_P_Peaks"]
                self.ecg_q_peaks[ecg, :, lead] = waves_peak["ECG_Q_Peaks"]
                self.ecg_s_peaks[ecg, :, lead] = waves_peak["ECG_S_Peaks"]
                df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 'p_mean'] = np.mean(self.ecg_p_peaks[ecg, :, lead])
                df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 'q_mean'] = np.mean(self.ecg_q_peaks[ecg, :, lead])
                df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 's_mean'] = np.mean(self.ecg_s_peaks[ecg, :, lead])
