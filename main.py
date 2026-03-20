import numpy as np
import neurokit2 as nk
import os
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum
import argparse
import warnings

columns = {
    'type': [],
    'ecg': [],
    'lead': [],
    'r_peak_mean': [],
    't_mean': [],
    'p_mean': [],
    'q_mean': [],
    's_mean': [],
    'rr_interval_mean': [],  # ms
    'qt_interval_mean': [],  # ms
    'pr_interval_mean': [],  # ms
}


def init(df):
    """
    Initialize the dataframe for the mean data points for the ECGs.
    :param df: The dataframe.
    :return: An initialized dataframe.
    """
    df['ecg'] = df['ecg'].astype(int)  # The ECG column should be integer and not float
    df.set_index(['type', 'ecg', 'lead'], inplace=True)  # Use the type, ECG number and lead number as the index
    return df


def create_output_dir():
    output_dir = 'output/'
    ecg_mean_data_dir = 'ecg_mean_data/'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + ecg_mean_data_dir, exist_ok=True)
    return output_dir + ecg_mean_data_dir


def save_dataframe(df, params):
    """
    Save the dataframe to csv file.
    :param df: The dataframe.
    :param params: The parameters' dictionary.
    :return: None
    """
    output_name = f'ecg_mean_data_ecg_{params['ecg_start']}_{params['ecg_end']}' \
                  f'_lead_{params['lead_start']}_{params['lead_end']}.csv'
    df.to_csv(create_output_dir() + output_name, index=True)


def order_csv_dataframe(params):
    """
    Reorder the csv dataframe so that it alternates between original and reconstructed ECGs.
    :param params: The parameters' dictionary.
    :return:
    """
    ecg_data_dir = create_output_dir()
    file = f'ecg_mean_data_ecg_{params['ecg_start']}_{params['ecg_end']}' \
           f'_lead_{params['lead_start']}_{params['lead_end']}.csv'
    assert os.path.isfile(ecg_data_dir + file), f'{file} does not exist'
    df = pd.read_csv(ecg_data_dir + file)

    len_org = (params['ecg_end'] - params['ecg_start']) * (params['lead_end'] - params['lead_start'])
    org_df = df[:len_org:]
    rec_df = df[len_org:]
    org_df['key'] = np.arange(len(org_df)) * 2
    rec_df['key'] = np.arange(len(rec_df)) * 2 + 1
    sorted_df = pd.concat([org_df, rec_df]).sort_values(by=['key']).drop(columns=['key'])
    sorted_df = sorted_df.reset_index(drop=True)
    sorted_df.to_csv(ecg_data_dir + 'ordered_' + file, index=False)


SAMPLING_RATE = 500  # Data is retrieved from records500
LEAD_LABELS = [
    'I', 'II', 'III',
    'aVR', 'aVL', 'aVF',
    'V1', 'V2', 'V3',
    'V4', 'V5', 'V6',
]
TOTAL_NUM_LEADS = len(LEAD_LABELS)
TOTAL_NUM_ECGS = 0  #15#4368


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
        # TODO: Maybe redefine the keys for the R-peaks to drop the slice() bit
        return self.ecg_r_peaks[(ecg_number, slice(None, None, None), lead_number)]

    def get_t_peaks(self, ecg_number: int, lead_number: int, t_value: int = None):
        self.check_boundaries(ecg_number, lead_number)
        if t_value is not None:
            return self.ecg_t_peaks[ecg_number, :, lead_number][t_value]
        return self.ecg_t_peaks[ecg_number, :, lead_number]

    def get_p_peaks(self, ecg_number: int, lead_number: int = None, p_value: int = None):
        self.check_boundaries(ecg_number, lead_number)
        if p_value is not None:
            return self.ecg_p_peaks[ecg_number, :, lead_number][p_value]
        return self.ecg_p_peaks[ecg_number, :, lead_number]

    def get_q_peaks(self, ecg_number: int, lead_number: int, q_value: int = None):
        self.check_boundaries(ecg_number, lead_number)
        if q_value is not None:
            return self.ecg_q_peaks[ecg_number, :, lead_number][q_value]
        return self.ecg_q_peaks[ecg_number, :, lead_number]

    def get_s_peaks(self, ecg_number: int, lead_number: int, s_value: int = None):
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

    def find_r_peaks(self, df, params):
        """
        Find R-peaks for a range of ECGs and leads and store the peaks and mean of peaks.
        :param df: The dataframe to save the r_peak mean values to.'
        :param params: The parameters' dictionary.
        :return: None
        """
        print(f'Calculating mean of R-peaks on {self.get_ecg_type()} ECG(s) [{params['ecg_start']} - '
              f'{params['ecg_end'] - 1}] using lead(s) [{LEAD_LABELS[params['lead_start']]} - '
              f'{LEAD_LABELS[params['lead_end'] - 1]}]')
        for ecg in range(params['ecg_start'], params['ecg_end']):
            for lead in range(params['lead_start'], params['lead_end']):
                peaks, r_peaks_top = self.get_r_peaks_from_signal(params, ecg_num=ecg, lead_num=lead)
                r_peak_top_mean = self.calculate_peak_mean(r_peaks_top, ecg, lead, print_r_peaks=False,
                                                           print_mean_values=False)
                df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 'r_peak_mean'] = r_peak_top_mean
                _, rr_interval_mean = self.calculate_rr_intervals(ecg=ecg, lead=lead, r_peaks=peaks,
                                                                  print_rr_intervals=False)
                df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 'rr_interval_mean'] = rr_interval_mean

    def calculate_rr_intervals(self, ecg, lead, r_peaks, print_rr_intervals=False):
        """
        Calculate the r-peak intervals for an ECG.
        :param ecg: The ECG.
        :param lead: The lead.
        :param r_peaks: The R-peaks for the ECG.
        :param print_rr_intervals: Print the RR-intervals and the RR-interval mean.
        :return:
        """
        #hrv = nk.hrv_time(r_peaks, sampling_rate=SAMPLING_RATE, show=True)
        #print(f'HRV RR-mean: {hrv["HRV_MeanNN"]}')
        rr_intervals = (np.diff(
            r_peaks) / SAMPLING_RATE) * 1000  # Multiply by 1000 to convert intervals to milliseconds
        rr_interval_mean = np.mean(rr_intervals)
        if print_rr_intervals:
            print(f'R-R intervals for {self.get_ecg_type()} ECG {ecg} [{LEAD_LABELS[lead]}]: {rr_intervals} ms')
            print(f'R-R interval mean for {self.get_ecg_type()} '
                  f'ECG {ecg} [{LEAD_LABELS[lead]}]: {rr_interval_mean:.4f} ms')
        return rr_intervals, rr_interval_mean

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
        ecg, signal, r_peaks, r_peak_tops = self.process_signal(ecg_num, lead_num, params)
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

    def process_signal(self, ecg_num, lead_num, params):
        """
        Process the ECG signal using NeuroKit's preprocessing pipeline.
        :param ecg_num: The index of the ECG signal.
        :param lead_num: The lead.
        :param params: The parameters' dictionary.
        :return: A processed ECG signal, its signal information and the R-peaks.
        """
        ecg = self.get_ecg(ecg_num, lead_num, params['process_method'])
        ecg, _ = nk.ecg_invert(ecg, sampling_rate=SAMPLING_RATE)
        if params['print_quality']:
            ecg_quality = nk.ecg_quality(ecg, sampling_rate=SAMPLING_RATE, method=params['quality_method'])
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

    def get_ecg(self, ecg_num: int, lead_num: int, process_method='pantompkins1985'):
        """
        Get and clean a single ECG signal.
        :param ecg_num: The index of the ECG signal.
        :param lead_num: The lead to plot.
        :param process_method: The process method.
        :return: ECG signal.

        Example: To get the first ECG with the first lead of the original data:
        ::
            ecg_signal = original.get_ecg(0, 0)
        """
        self.check_boundaries(ecg_num, lead_num)
        ecg = self.data[ecg_num, :, lead_num]
        cleaned_ecg = nk.ecg_clean(ecg, sampling_rate=SAMPLING_RATE, method=process_method)
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
        is_first = True
        for ecg in range(params['ecg_start'], params['ecg_end']):
            for lead in range(params['lead_start'], params['lead_end']):
                current_ecg = self.get_ecg(ecg_num=ecg, lead_num=lead, process_method=params['process_method'])
                r_peaks = self.get_r_peaks(ecg, lead)
                if params['use_show']:
                    _, waves = nk.ecg_delineate(current_ecg, r_peaks, sampling_rate=SAMPLING_RATE,
                                                     method=params['delineate_method'], show=True, show_type='all')
                    plt.grid(True, alpha=0.3)
                    plt.title(f'Delineated {self.get_ecg_type()} ECG {ecg} - lead {LEAD_LABELS[lead]}')
                    plt.show()
                else:
                    _, waves = nk.ecg_delineate(current_ecg, r_peaks, sampling_rate=SAMPLING_RATE,
                                                     method=params['delineate_method'], show=False, show_type='all')
                if params['print_keys'] and is_first:
                    print(f'Keys in waves dictionary: {sorted(waves.keys())}')
                is_first = False
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
                if params['print_peaks']:
                    print(f'T-peaks for {self.get_ecg_type()} ECG {ecg} and '
                          f'lead {LEAD_LABELS[lead]}: {waves["ECG_T_Peaks"]}')
                    print(f'P-peaks for {self.get_ecg_type()} ECG {ecg} and '
                          f'lead {LEAD_LABELS[lead]}: {waves["ECG_P_Peaks"]}')
                    print(f'Q-peaks for {self.get_ecg_type()} ECG {ecg} and '
                          f'lead {LEAD_LABELS[lead]}: {waves["ECG_Q_Peaks"]}')
                    print(f'S-peaks for {self.get_ecg_type()} ECG {ecg} and '
                          f'lead {LEAD_LABELS[lead]}: {waves["ECG_S_Peaks"]}')
                self.value_mean_t(df, ecg, lead, current_ecg, waves, print_exception=False)
                self.value_mean_p(df, ecg, lead, current_ecg, waves, print_exception=False)
                self.value_mean_q(df, ecg, lead, current_ecg, waves, print_exception=False)
                self.value_mean_s(df, ecg, lead, current_ecg, waves, print_exception=False)
                self.calculate_qt_intervals(df, ecg, lead, waves["ECG_R_Onsets"],
                                            waves['ECG_T_Offsets'], params['print_mean'])
                self.calculate_pr_intervals(df, ecg, lead, waves["ECG_P_Onsets"],
                                            waves['ECG_R_Onsets'])

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

    def value_mean_t(self, df, ecg, lead, current_ecg, waves_peak, print_exception=False):
        try:
            self.ecg_t_peaks[ecg, :, lead] = waves_peak["ECG_T_Peaks"]
            t_peak_tops = current_ecg[self.ecg_t_peaks[ecg, :, lead]]
            t_peak_top_mean = self.calculate_peak_mean(t_peak_tops, ecg, lead)
            df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 't_mean'] = t_peak_top_mean
        except IndexError as e:
            if print_exception:
                print(f'Error finding T-peak mean for {self.get_ecg_type()} ECG {ecg} and lead {lead}:\n\t{e}')

    def value_mean_p(self, df, ecg, lead, current_ecg, waves_peak, print_exception=False):
        try:
            self.ecg_p_peaks[ecg, :, lead] = waves_peak["ECG_P_Peaks"]
            p_peak_tops = current_ecg[self.ecg_p_peaks[ecg, :, lead]]
            p_peak_top_mean = self.calculate_peak_mean(p_peak_tops, ecg, lead)
            df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 'p_mean'] = p_peak_top_mean
        except IndexError as e:
            if print_exception:
                print(f'Error finding P-peaks for {self.get_ecg_type()} ECG {ecg} and lead {lead}:\n\t{e}')

    def value_mean_q(self, df, ecg, lead, current_ecg, waves_peak, print_exception=False):
        try:
            self.ecg_q_peaks[ecg, :, lead] = waves_peak["ECG_Q_Peaks"]
            q_peak_tops = current_ecg[self.ecg_q_peaks[ecg, :, lead]]
            q_peak_top_mean = self.calculate_peak_mean(q_peak_tops, ecg, lead)
            df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 'q_mean'] = q_peak_top_mean
        except IndexError as e:
            if print_exception:
                print(f'Error finding Q-peaks for {self.get_ecg_type()} ECG {ecg} and lead {lead}:\n\t{e}')

    def value_mean_s(self, df, ecg, lead, current_ecg, waves_peak, print_exception=False):
        try:
            self.ecg_s_peaks[ecg, :, lead] = waves_peak["ECG_S_Peaks"]
            s_peak_tops = current_ecg[self.ecg_s_peaks[ecg, :, lead]]
            s_peak_top_mean = self.calculate_peak_mean(s_peak_tops, ecg, lead)
            df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 's_mean'] = s_peak_top_mean
        except IndexError as e:
            if print_exception:
                print(f'Error finding S-peaks for {self.get_ecg_type()} ECG {ecg} and lead {lead}:\n\t{e}')


def set_parameters(params, args):
    """
    Set the parameters' dict using the command line arguments.
    :param params:
    :param args:
    :return:
    """
    params['ecg_start'] = args.ecg_start
    params['ecg_end'] = args.ecg_end
    params['lead_start'] = args.lead_start
    params['lead_end'] = args.lead_end
    params['use_plotting'] = args.use_plotting
    params['print_peaks'] = args.print_peaks
    params['use_show'] = args.use_show
    params['zoom'] = args.zoom
    params['zoom_level'] = args.zoom_level
    params['print_quality'] = args.print_quality
    params['use_segment'] = args.use_segment
    params['print_mean'] = args.print_mean
    params['save_csv'] = args.save_csv
    params['delineate_method'] = args.delineate_method
    params['process_method'] = args.process_method
    params['quality_method'] = args.quality_method
    params['print_keys'] = args.print_keys


def parse_arguments():
    """
    Parse command line arguments
    :return:
    """
    parser = argparse.ArgumentParser(prog=__name__, description='Reconstruction of 12-lead from digitized paper ECG')
    parser.add_argument('--ecg_start', default=0, type=int, help='The starting ECG index')
    parser.add_argument('--ecg_end', default=TOTAL_NUM_ECGS, type=int, help='The ending ECG index')
    parser.add_argument('--lead_start', default=0, type=int, help='The starting lead index')
    parser.add_argument('--lead_end', default=TOTAL_NUM_LEADS, type=int, help='The ending lead index')
    parser.add_argument('--use_plotting', action='store_true', help='Whether to use plotting')
    parser.add_argument('--print_peaks', action='store_true', help='Use print peaks')
    parser.add_argument('--use_show', action='store_true', help='Use the show method from neurokit2')
    parser.add_argument('--zoom', action='store_true', help='Use zoom when plotting')
    parser.add_argument('--zoom_level', default=5, type=int, help='The level of zoom when plotting')
    parser.add_argument('--print_quality', action='store_true', help='Print the ECG quality')
    parser.add_argument('--use_segment', action='store_true', help='Show ECGs as segments')
    parser.add_argument('--print_mean', action='store_true',
                        help='Print the mean values for peaks and intervals')
    parser.add_argument('--save_csv', action='store_true', help='Save the ECGs to a csv file')
    parser.add_argument('--delineate_method', type=str,
                        choices=['cwt', 'dwt'], default='cwt', help='Which delineate method to use')
    parser.add_argument('--process_method', type=str,
                        choices=['neurokit', 'pantompkins1985'], default='pantompkins1985',
                        help='Which process method to use')
    parser.add_argument('--quality_method', type=str,
                        choices=['averageQRS', 'templatematch'], default='templatematch',
                        help='Which quality method to use')
    parser.add_argument('--print_keys', action='store_true',
                        help='After delineating print the keys in the waves dict')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    arguments = parse_arguments()
    parameters = {}
    set_parameters(parameters, arguments)

    ecg_data = init(pd.DataFrame(columns))
    # TODO: Make a container for the data so we don't have to load it twice
    original = ECG('output/data.npy.npz', TypeECG.ORIGINAL)
    reconstructed = ECG('output/data.npy.npz', TypeECG.RECONSTRUCTED)
    assert original.__eq__(reconstructed), "Original and reconstructed signals do not match."

    original.find_r_peaks(ecg_data, parameters)
    original.find_ecg_peaks(ecg_data, parameters)

    reconstructed.find_r_peaks(ecg_data, parameters)
    reconstructed.find_ecg_peaks(ecg_data, parameters)
    if parameters['save_csv']:
        save_dataframe(ecg_data, parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore SettingWithCopyWarning warnings
            order_csv_dataframe(parameters)
