import json
import math
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import os
import neurokit2 as nk
from enum import Enum
import pandas as pd
from util import *
import warnings


class TypeECG(Enum):
    ORIGINAL = 1
    RECONSTRUCTED = 2


class ECG:
    def __init__(self, input_data: Any, type_ecg: TypeECG, data_column: str=None):
        """

        :param input_data: The input file, either .npy or .npz.
        :param type_ecg: If it's original or reconstructed data.
        :param data_column: If the input data has multiple columns, such as if it contains both the
            original and reconstructed data, the column to use must be provided. If the input data
            only has one column, this parameter can be ignored.
        """
        self.type_ecg = type_ecg
        self.data = self.load_data(input_data, data_column)
        self.ecg_r_peaks = {}


    def get_r_peaks_empty(self):
        """
        Check if the ecg_r_peaks dictionary is empty since certain operations requires it to be not empty.
        :return: True if empty, False otherwise.
        """
        if not self.ecg_r_peaks:
            return True
        return False

    def get_r_peaks(self, ecg_number: int, lead_number: int, r_value: int = None):
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
        return self.ecg_r_peaks[(ecg_number, slice(None, None, None), lead_number)]

    def get_shape(self):
        return self.data.shape

    def __eq__(self, other) -> bool | Exception:
        """
        Compare the shape of the data as both the original and reconstructed data should have the same shape.
        The shape is the number of ECGs, leads and samples for the ECG objects.
        :param other: The other ECG object.
        :return:
        """
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.get_shape() == other.get_shape()

    @staticmethod
    def load_data(data: str, data_column: str=None) -> Any | None:
        """
        Load data from a .npy or .npz file. If you know the name of the key with the data it can be supplied
        when creating the ECG object, otherwise it must be supplied manually in the function.
        :param data: The file with data.
        :param data_column: The column name of the data. Required if the file contains multiple keys.
        :return: The original or reconstructed data.
        """
        if not os.path.exists(data):
            raise FileNotFoundError(f'File {data} not found.')
        if data.endswith('.npy'):
            return np.load(data)
        if data.endswith('.npz'):
            with np.load(data) as data:
                try:
                    if data_column is not None:
                        v = data[data_column]
                    else:
                        keys = data.files
                        if len(keys) > 1:
                            print(f'File {data} has multiple keys: {keys}, '
                                  'the key needs to be specified when creating the ECG object.')
                            raise KeyError
                        else:
                            v = data[keys[0]]
                except KeyError:
                    raise KeyError(f'File {data} has the key(s): {data.keys()}')
            return v
        print(f'File {data} is of an unrecognized format.')
        return None

    def find_r_peaks(self, df: pd.DataFrame, params: dict[str, Any], **kwargs) -> None:
        """
        Find R-peaks for a range of ECGs and leads and store the peaks and mean of peaks.
        :param df: The dataframe to save the r_peak mean values to.'
        :param params: The parameters' dictionary.
        :return: None
        """
        print(f'Calculating mean of R-peaks on {self.get_ecg_type()} ECG(s) [{params['ecg_start']} - '
              f'{params['ecg_end'] - 1}] using leads {', '.join(LEAD_LABELS)}')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for ecg in range(params['ecg_start'], params['ecg_end']):
                for lead in range(params['lead_start'], params['lead_end']):
                    try:
                        peaks, r_peaks_top = self.get_r_peaks_from_signal(params, ecg_num=ecg, lead_num=lead)
                        r_peak_top_mean = self.calculate_peak_mean(r_peaks_top, ecg, lead, print_r_peaks=False,
                                                                   print_mean_values=False)
                        df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 'r_peak_mean'] = r_peak_top_mean
                        _, rr_interval_mean = self.calculate_rr_intervals(ecg=ecg, lead=lead, r_peaks=peaks,
                                                                          print_rr_intervals=False)
                        df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 'rr_interval_mean'] = rr_interval_mean
                        df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 'success'] = 1
                    except (ValueError, ZeroDivisionError) as e:
                        if 'print_exception' in kwargs:
                            if kwargs['print_exception']:
                                print(f'Error finding R-peaks for ECG {ecg} lead {LEAD_LABELS[lead]}: {e}')
                        df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 'success'] = 0
                        continue
                if ecg % 100 == 0:
                    print(f'Processed {ecg} ECGs out of {TOTAL_NUM_ECGS} ECGs ({round((ecg/TOTAL_NUM_ECGS)*100)}%)')

    def calculate_rr_intervals(self, ecg, lead, r_peaks, print_rr_intervals=False):
        """
        Calculate the r-peak intervals for an ECG.
        :param ecg: The ECG.
        :param lead: The lead.
        :param r_peaks: The R-peaks for the ECG.
        :param print_rr_intervals: Print the RR-intervals and the RR-interval mean.
        :return:
        """
        hrv = nk.hrv_time(r_peaks, sampling_rate=SAMPLING_RATE, show=False)
        rr_interval_mean = np.mean(hrv["HRV_MeanNN"])
        if print_rr_intervals:
            print(f'R-R intervals for {self.get_ecg_type()} ECG {ecg} [{LEAD_LABELS[lead]}]: {hrv["HRV_MeanNN"]} ms')
            print(f'R-R interval mean for {self.get_ecg_type()} '
                  f'ECG {ecg} [{LEAD_LABELS[lead]}]: {rr_interval_mean:.4f} ms')
        return hrv["HRV_MeanNN"], rr_interval_mean

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

    def get_r_peaks_from_signal(self, params, ecg_num, lead_num):
        """
        Get the R-peaks from an ECG and lead signal.
        :param params: The parameters' dictionary.
        :param ecg_num: The ECG index.
        :param lead_num: The lead
        :return: The R-peaks from the signal.
        """
        ecg, signal, r_peaks, r_peak_tops = self.process_signal(ecg_num, lead_num, params, print_quality=False)
        self.ecg_r_peaks[ecg_num, :, lead_num] = r_peaks["ECG_R_Peaks"]
        if self.type_ecg == TypeECG.ORIGINAL:
            title = '- Original '
        else:
            title = '- Reconstructed '
        peaks = self.plot_r_peaks(ecg, r_peaks, params,
                                  title_suffix=title + f'- ECG {ecg_num} Lead {LEAD_LABELS[lead_num]}')
        return peaks, r_peak_tops

    @staticmethod
    def plot_r_peaks(ecg_signal, info, params, title_suffix=''):
        """
        After getting the R-peaks, the R-peaks can be plotted.
        :param ecg_signal: The ECG signal.
        :param info: The ECG information.
        :param params: The parameters' dictionary.
        :param title_suffix: The plot title.
        :return: R-peaks.
        """
        rpeaks = info['ECG_R_Peaks']
        if params['use_plotting']:
            if params['zoom']:
                max_idx = int(params['zoom_level'] * SAMPLING_RATE)
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

    def process_signal(self, ecg_num: int, lead_num: int, params: dict[str, Any], **kwargs):
        """
        Process the ECG signal using NeuroKit's preprocessing pipeline.
        :param ecg_num: The index of the ECG signal.
        :param lead_num: The lead.
        :param params: The parameters' dictionary.
        :return: A processed ECG signal, its signal information and the R-peaks.
        """
        ecg = self.get_ecg(ecg_num, lead_num)
        ecg = nk.ecg_clean(ecg, sampling_rate=SAMPLING_RATE, method='neurokit')
        ecg, _ = nk.ecg_invert(ecg, sampling_rate=SAMPLING_RATE)
        if 'print_quality' in kwargs:
            if kwargs['print_quality']:
                if 'print_quality_method' in kwargs:
                    ecg_quality = nk.ecg_quality(ecg, sampling_rate=SAMPLING_RATE, method=kwargs['print_quality_method'])
                else:
                    ecg_quality = nk.ecg_quality(ecg, sampling_rate=SAMPLING_RATE, method='templatematch')
                num = f'{self.get_ecg_type()} ECG {ecg_num} {LEAD_LABELS[lead_num]}'
                print(f'{num} quality (mean±std): {np.mean(ecg_quality):.3f} ± {np.std(ecg_quality):.3f}')
        signals, r_peaks = nk.ecg_process(ecg, sampling_rate=SAMPLING_RATE, method=params['process_method'])
        if params['use_segment']:
            nk.ecg_segment(ecg, sampling_rate=SAMPLING_RATE, show=True)
            x = y = 0.02
            plt.figtext(x, y, f'{self.get_ecg_type()} ECG {ecg_num} Lead {LEAD_LABELS[lead_num]}')
            plt.grid(True, alpha=0.3)
            plt.show()
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
        return self.data[ecg_num, :, lead_num]

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
        Print the ECG type where only the first letter is capitalized.
        :return:
        """
        return self.type_ecg.name[0] + self.type_ecg.name[1:].lower()

    def rr_to_bpm(self, df, ecg_num=0, lead_num=0):
        """
        Print the mean heart rate using the RR-mean for an ECG signal.
        :param df: The dataframe with the RR-interval data.
        :param ecg_num: The ECG signal index.
        :param lead_num: The lead.
        :return: None
        """
        if self.type_ecg == TypeECG.ORIGINAL:
            row = df.loc[('Original', ecg_num, LEAD_LABELS[lead_num])]
        else:
            row = df.loc[('Reconstructed', ecg_num, LEAD_LABELS[lead_num])]
        rr_ms = float(row['rr_interval_mean'])
        hr_bpm = 60000.0 / rr_ms
        if hr_bpm < 0.0:
            hr_bpm = float('nan')
        print(f'{self.get_ecg_type()} ECG {ecg_num} lead {LEAD_LABELS[lead_num]}: '
              f'RR mean: {rr_ms:.1f} ms - HR mean: {hr_bpm:.1f} BPM')

    def find_ecg_peaks(self, df, params):
        """
        Find the peaks of the ECG signals.
        :param df: The dataframe to save the ECG peaks to.
        :param params: The parameters' dictionary.
        :return: None
        """
        if self.get_r_peaks_empty():
            print('Finding the ECG peaks requires having the R-peaks, find the R-peaks before the ECG peaks.')
            raise SystemExit
        print(f'Finding peaks on {self.get_ecg_type()} ECG(s) [{params['ecg_start']} - '
              f'{params['ecg_end'] - 1}] using leads {', '.join(LEAD_LABELS)}')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for ecg in range(params['ecg_start'], params['ecg_end']):
                for lead in range(params['lead_start'], params['lead_end']):
                    current_ecg = self.get_ecg(ecg_num=ecg, lead_num=lead)
                    if not df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 'success']:
                        continue
                    r_peaks = self.get_r_peaks(ecg, lead)
                    try:
                        if params['use_show']:
                            _, waves = nk.ecg_delineate(current_ecg, r_peaks, sampling_rate=SAMPLING_RATE,
                                                        method=params['delineate_method'], show=True, show_type='all')
                            plt.grid(True, alpha=0.3)
                            plt.title(f'Delineated {self.get_ecg_type()} ECG {ecg} - lead {LEAD_LABELS[lead]}')
                            plt.show()
                            plt.close()
                        else:
                            _, waves = nk.ecg_delineate(current_ecg, r_peaks, sampling_rate=SAMPLING_RATE,
                                                        method=params['delineate_method'], show=False, show_type='all')
                    except Exception as e:
                        print(f'Invalid ECG signal index {ecg} {LEAD_LABELS[lead]} out of range: {e}')
                        continue
                    if params['use_plotting']:
                        nk.events_plot([waves['ECG_T_Peaks'][:params['zoom_level']],
                                        waves['ECG_P_Peaks'][:params['zoom_level']],
                                        waves['ECG_Q_Peaks'][:params['zoom_level']],
                                        waves["ECG_R_Onsets"][:params['zoom_level']],
                                        waves['ECG_S_Peaks'][:params['zoom_level']]],
                                       current_ecg[:(params['zoom_level'] + 1) * SAMPLING_RATE])
                        plt.grid(True, alpha=0.3)
                        plt.title(f'Peaks - {self.get_ecg_type()} ECG - {ecg} - Lead {LEAD_LABELS[lead]}')
                        plt.xlabel('Samples')
                        plt.ylabel('Amplitude')
                        plt.tight_layout()
                        plt.show()
                        plt.close()
                    if params['print_peaks']:
                        print(f'T-peaks for {self.get_ecg_type()} ECG {ecg} '
                              f'lead {LEAD_LABELS[lead]}: {waves["ECG_T_Peaks"]}')
                        print(f'P-peaks for {self.get_ecg_type()} ECG {ecg} '
                              f'lead {LEAD_LABELS[lead]}: {waves["ECG_P_Peaks"]}')
                        print(f'Q-peaks for {self.get_ecg_type()} ECG {ecg} '
                              f'lead {LEAD_LABELS[lead]}: {waves["ECG_Q_Peaks"]}')
                        print(f'S-peaks for {self.get_ecg_type()} ECG {ecg} '
                              f'lead {LEAD_LABELS[lead]}: {waves["ECG_S_Peaks"]}')
                    self.value_mean_t(df, ecg, lead, current_ecg, waves, print_exception=False)
                    self.value_mean_p(df, ecg, lead, current_ecg, waves, print_exception=False)
                    self.value_mean_q(df, ecg, lead, current_ecg, waves, print_exception=False)
                    self.value_mean_s(df, ecg, lead, current_ecg, waves, print_exception=False)
                    self.calculate_qt_intervals(df, ecg, lead, waves["ECG_R_Onsets"],
                                                waves['ECG_T_Offsets'], params['print_mean'])
                    self.calculate_pr_intervals(df, ecg, lead, waves["ECG_P_Onsets"],
                                                waves['ECG_R_Onsets'])
                    df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 'ecg_signal'] = json.dumps(current_ecg.tolist())
                    df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 'r_peaks'] = json.dumps(r_peaks.tolist())
                if ecg % 100 == 0:
                    print(f'Processed {ecg} ECG peaks out of {TOTAL_NUM_ECGS} ECGs ({int((ecg/TOTAL_NUM_ECGS)*100)}%)')

    @staticmethod
    def convert_array(x):
        """
        The list data retrieved from the delineate function needs to be converted to a numpy array for finding intervals.
        :param x:
        :return:
        """
        arr = np.asarray(x, dtype=float)
        return arr.ravel()

    def validate_data(self, data1, data2):
        """
        When finding intervals, the interval columns must be converted to numpy arrays. Then they are made equal length
        so that each value in the arrays can be compared. Finally, the arrays are checked for NaN values and all
        non-NaN values are ignored.
        :param data1:
        :param data2:
        :return:
        """
        data1 = self.convert_array(data1)
        data2 = self.convert_array(data2)
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]

        valid_mask = (~np.isnan(data1)) & (~np.isnan(data2))
        samples = data2[valid_mask] - data1[valid_mask]
        return samples

    def calculate_qt_intervals(self, df, ecg, lead, q_onsets, t_offsets, print_mean):
        """
        Calculate the QT intervals using the QRS-onset and T-offset.
        :param df:
        :param ecg:
        :param lead:
        :param q_onsets:
        :param t_offsets:
        :param print_mean:
        :return:
        """
        samples = self.validate_data(q_onsets, t_offsets)
        qt_ms = (samples / SAMPLING_RATE) * 1000  # Convert to milliseconds
        qt_mean = np.mean(qt_ms)
        if print_mean:
            print(f'QT intervals for {self.get_ecg_type()} ECG {ecg} {LEAD_LABELS[lead]} {qt_ms} (ms):')
            print(f'QT mean for {self.get_ecg_type()} ECG {ecg} {LEAD_LABELS[lead]}: {qt_mean:.4f} ms')
        df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 'qt_interval_mean'] = qt_mean

    def calculate_pr_intervals(self, df, ecg, lead, p_onsets, q_onsets):
        """
        Calculate the PR intervals using the P-onset and QRS-onset.
        :param df:
        :param ecg:
        :param lead:
        :param p_onsets:
        :param q_onsets:
        :return:
        """
        pr_ms = self.pair_pr_intervals(p_onsets, q_onsets, SAMPLING_RATE)
        pr_mean = float(np.nanmean(pr_ms)) if pr_ms.size else np.nan
        df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 'pr_interval_mean'] = pr_mean

    def pair_pr_intervals(self, p_onsets, q_onsets, fs, min_ms=100.0, max_ms=240.0):
        """

        :param p_onsets:
        :param q_onsets:
        :param fs:
        :param min_ms:
        :param max_ms:
        :return:
        """
        p_arr = self.convert_array(p_onsets); p_arr = p_arr[~np.isnan(p_arr)]
        q_arr = self.convert_array(q_onsets); q_arr = q_arr[~np.isnan(q_arr)]
        if p_arr.size == 0 or q_arr.size == 0:
            return np.array([], dtype=float)
        min_gap = int((min_ms/1000.0)*fs)
        max_gap = int((max_ms/1000.0)*fs)
        pr_samples = []
        j = 0
        for pi in p_arr:
            while j < q_arr.size and q_arr[j] < pi:
                j += 1
            if j >= q_arr.size:
                break
            gap = q_arr[j] - pi
            if min_gap <= gap <= max_gap:
                pr_samples.append(gap)
        return (np.asarray(pr_samples, dtype=float) / fs) * 1000.0


    @staticmethod
    def nan_threshold(peaks, threshold=0.20) -> list[int]:
        """
        If an ECG has any NaNs in the T, P, Q or S column it will lead to NaN in the df/db, so we
        check for NaN and if it's below a threshold they are dropped, otherwise we keep it in.
        :param peaks: The list of peaks.
        :param threshold: The threshold to consider, 20% by default. Must be between 0 and 1.
        :return:
        """
        if not 0 <= threshold <= 1:
            print(f'Threshold must be between 0 and 1, but got {threshold}, defaulting to 0.20')
            threshold = 0.20
        if np.isnan(peaks).any():
            nan_count = sum(math.isnan(x) for x in peaks if isinstance(x, float))
            if nan_count / len(peaks) < threshold:
                peaks = [x for x in peaks if x == x]
        return peaks

    def value_mean_t(self, df, ecg, lead, current_ecg, waves_peak, print_exception=False):
        try:
            peaks = waves_peak["ECG_T_Peaks"]
            peaks = self.nan_threshold(peaks)
            t_peak_tops = current_ecg[peaks]
            t_peak_top_mean = self.calculate_peak_mean(t_peak_tops, ecg, lead)
            df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 't_mean'] = t_peak_top_mean
        except IndexError as e:
            if print_exception:
                print(f'Error finding T-peak mean for {self.get_ecg_type()} ECG {ecg} lead {LEAD_LABELS[lead]}:\n\t{e}')

    def value_mean_p(self, df, ecg, lead, current_ecg, waves_peak, print_exception=False):
        try:
            peaks = waves_peak["ECG_P_Peaks"]
            peaks = self.nan_threshold(peaks)
            p_peak_tops = current_ecg[peaks]
            p_peak_top_mean = self.calculate_peak_mean(p_peak_tops, ecg, lead)
            df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 'p_mean'] = p_peak_top_mean
        except IndexError as e:
            if print_exception:
                print(f'Error finding P-peaks for {self.get_ecg_type()} ECG {ecg} lead {LEAD_LABELS[lead]}:\n\t{e}')

    def value_mean_q(self, df, ecg, lead, current_ecg, waves_peak, print_exception=False):
        try:
            peaks = waves_peak["ECG_Q_Peaks"]
            peaks = self.nan_threshold(peaks)
            q_peak_tops = current_ecg[peaks]
            q_peak_top_mean = self.calculate_peak_mean(q_peak_tops, ecg, lead)
            df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 'q_mean'] = q_peak_top_mean
        except IndexError as e:
            if print_exception:
                print(f'Error finding Q-peaks for {self.get_ecg_type()} ECG {ecg} lead {LEAD_LABELS[lead]}:\n\t{e}')

    def value_mean_s(self, df, ecg, lead, current_ecg, waves_peak, print_exception=False):
        try:
            peaks = waves_peak["ECG_S_Peaks"]
            peaks = self.nan_threshold(peaks)
            s_peak_tops = current_ecg[peaks]
            s_peak_top_mean = self.calculate_peak_mean(s_peak_tops, ecg, lead)
            df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 's_mean'] = s_peak_top_mean
        except IndexError as e:
            if print_exception:
                print(f'Error finding S-peaks for {self.get_ecg_type()} ECG {ecg} lead {LEAD_LABELS[lead]}:\n\t{e}')
