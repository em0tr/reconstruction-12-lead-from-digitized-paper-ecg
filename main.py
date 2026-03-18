import numpy as np
import neurokit2 as nk
import os
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum
import argparse

columns = {
    'type': [],
    'ecg': [],
    'lead': [],
    'r_peak_mean': [],
    't_mean': [],
    'p_mean': [],
    'q_mean': [],
    's_mean': [],
    'rr_interval_mean': [],
    'qt_interval_mean': [],
}

def init(df):
    """
    Initialize the dataframe for the mean data points for the ECGs.
    :param df: The dataframe.
    :return: An initialized dataframe.
    """
    df['ecg'] = df['ecg'].astype(int) # The ECG column should be integer and not float
    df.set_index(['type', 'ecg', 'lead'], inplace=True) # Use the type, ECG number and lead number as the index
    return df


def create_output_dir():
    output_dir = 'output/'
    ecg_mean_data_dir = 'ecg_mean_data/'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + ecg_mean_data_dir, exist_ok=True)
    return output_dir + ecg_mean_data_dir


def save_dataframe(df, ecg_start, ecg_end, lead_start, lead_end, use_org=False, use_rec=False):
    """
    Save the dataframe to csv file.
    :param df: The dataframe.
    :param ecg_start: The start index of the ECG.
    :param ecg_end: The end index of the ECG.
    :param lead_start: The start index of the lead.
    :param lead_end: The end index of the lead.
    :param use_org: Use only original data in the dataframe. Can be used to save only original data.
    :param use_rec: Use only reconstructed data in the dataframe. Can be used to save only reconstructed data.
    :return: None
    """
    if use_org and not use_rec:
        output_name = f'ecg_mean_data_original_ecg_{ecg_start}_{ecg_end}_lead_{lead_start}_{lead_end}.csv'
    elif use_org and not use_rec:
        output_name = f'ecg_mean_data_reconstructed_ecg_{ecg_start}_{ecg_end}_lead_{lead_start}_{lead_end}.csv'
    else:
        output_name = f'ecg_mean_data_ecg_{ecg_start}_{ecg_end}_lead_{lead_start}_{lead_end}.csv'
    df.to_csv(create_output_dir() + output_name, index=True)


def order_csv_dataframe(ecg_start, ecg_end, lead_start, lead_end):
    """
    Reorder the csv dataframe so that it alternates between original and reconstructed ECGs.
    :param ecg_start:
    :param ecg_end:
    :param lead_start:
    :param lead_end:
    :return:
    """
    ecg_data_dir = create_output_dir()
    file = f'ecg_mean_data_ecg_{ecg_start}_{ecg_end}_lead_{lead_start}_{lead_end}.csv'
    assert os.path.isfile(ecg_data_dir + file), f'{file} does not exist'
    df = pd.read_csv(ecg_data_dir + file)

    len_org = (ecg_end - ecg_start) * (lead_end - lead_start)
    org_df = df[:len_org:]
    rec_df = df[len_org:]
    org_df['key'] = np.arange(len(org_df))*2
    rec_df['key'] = np.arange(len(rec_df))*2+1
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
                peaks, r_peaks_top = self.get_r_peaks_from_signal(ecg_num=ecg, lead_num=lead)
                r_peak_top_mean = self.calculate_peak_mean(r_peaks_top, ecg, lead, print_r_peaks=False, print_mean_values=False)
                df.loc[(self.get_ecg_type(), ecg, LEAD_LABELS[lead]), 'r_peak_mean'] = r_peak_top_mean
                _, rr_interval_mean = self.calculate_rr_intervals(ecg=ecg, lead=lead, r_peaks=peaks, print_rr_intervals=False)
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
        rr_intervals = (np.diff(r_peaks) / SAMPLING_RATE) * 1000 # Multiply by 1000 to convert intervals to milliseconds
        rr_interval_mean = np.mean(rr_intervals)
        if print_rr_intervals:
            print(f'R-R intervals for {self.get_ecg_type()} ECG {ecg} [{LEAD_LABELS[lead]}]: {rr_intervals} ms')
            print(f'R-R interval mean for {self.get_ecg_type()} ECG {ecg} [{LEAD_LABELS[lead]}]: {rr_interval_mean:.4f} ms')
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

    def get_r_peaks_from_signal(self, ecg_num, lead_num):
        """
        Get the R-peaks from an ECG and lead signal.
        :param ecg_num: The ECG index.
        :param lead_num: The lead
        :return: The R-peaks from the signal.
        """
        ecg, signal, r_peaks, r_peak_tops = self.process_signal(ecg_num, lead_num, use_segment=True, print_quality=False)
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

    def process_signal(self, ecg_num, lead_num, use_segment=False, print_quality=False):
        """
        Process the ECG signal using NeuroKit's preprocessing pipeline.
        :param ecg_num: The index of the ECG signal.
        :param lead_num: The lead.
        :param use_segment: Segment the signal to a single heartbeat.
        :param print_quality: Use the ecg_quality function from neurokit2 to assess the quality of the signal.
        :return: A processed ECG signal, its signal information and the R-peaks.
        """
        ecg = self.get_ecg(ecg_num, lead_num)
        ecg, _ = nk.ecg_invert(ecg, sampling_rate=SAMPLING_RATE)
        if print_quality:
            ecg_quality = nk.ecg_quality(ecg, sampling_rate=SAMPLING_RATE, method='templatematch')
            print(f'ECG quality (mean±std): {np.mean(ecg_quality):.3f} ± {np.std(ecg_quality):.3f}')
        signals, r_peaks = nk.ecg_process(ecg, sampling_rate=SAMPLING_RATE, method='pantompkins1985')
        if use_segment:
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
        ecg = self.data[ecg_num, :, lead_num]
        cleaned_ecg = nk.ecg_clean(ecg, sampling_rate=SAMPLING_RATE, method='pantompkins1985')
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

    def rr_to_bpm(self, df, ecg_num=0, lead_num=0, print_row=False):
        """
        Print the mean heart rate using the RR-mean for an ECG signal.
        :param df: The dataframe with the RR-interval data.
        :param ecg_num: The ECG signal index.
        :param lead_num: The lead.
        :param print_row: Print more of the dataframe row.
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
        if print_row:
            print(row)

    def find_ecg_peaks(self, df,
                       ecg_start=0, ecg_end=TOTAL_NUM_ECGS,
                       lead_start=0, lead_end=TOTAL_NUM_LEADS,
                       use_plotting=False, zoom_level=4,
                       print_peaks=False, use_show=False
                       ):
        """
        Find the peaks of the ECG signals.
        :param df: The dataframe to save the ECG peaks to.
        :param ecg_start: The start index of the ECG signal.
        :param ecg_end:  The end index of the ECG signal.
        :param lead_start: The start index of the lead.
        :param lead_end: The end index of the lead.
        :param use_plotting: Plot the ECG peaks.
        :param zoom_level: The zoom level for the plotted ECG peaks.
        :param print_peaks: Print the different ECG peaks.
        :param use_show: Show the plot from the ecg_delineate function.
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
                                                     method='cwt', show=True, show_type='all')
                    plt.grid(True, alpha=0.3)
                    plt.title(f'Delineated {self.get_ecg_type()} ECG {ecg} - lead {LEAD_LABELS[lead]}')
                    plt.show()
                else:
                    _, waves_peak = nk.ecg_delineate(current_ecg, r_peaks, sampling_rate=SAMPLING_RATE,
                                                     method='cwt', show=False, show_type='peaks')
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
                self.value_mean_t(df, ecg, lead, current_ecg, waves_peak, print_exception=False)
                self.value_mean_p(df, ecg, lead, current_ecg, waves_peak, print_exception=False)
                self.value_mean_q(df, ecg, lead, current_ecg, waves_peak, print_exception=False)
                self.value_mean_s(df, ecg, lead, current_ecg, waves_peak, print_exception=False)

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


parameters = {

}


def set_parameters(params, args):
    params['ecg_start'] = args.ecg_start
    params['ecg_end'] = args.ecg_end
    params['lead_start'] = args.lead_start
    params['lead_end'] = args.lead_end
    params['use_plotting'] = args.use_plotting
    params['print_peaks'] = args.print_peaks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=__name__, description=__doc__)
    parser.add_argument('--ecg_start', default=0, type=int, help='The starting ECG index')
    parser.add_argument('--ecg_end', default=TOTAL_NUM_ECGS, type=int, help='The ending ECG index')
    parser.add_argument('--lead_start', default=0, type=int, help='The starting lead index')
    parser.add_argument('--lead_end', default=TOTAL_NUM_LEADS, type=int, help='The ending lead index')
    parser.add_argument('--use_plotting', type=bool, default=False, help='Whether to use plotting or not')
    parser.add_argument('--print_peaks', type=bool, default=False, help='Use print peaks')
    arguments = parser.parse_args()
    print(arguments)
    set_parameters(parameters, arguments)
    print(parameters)

    ecg_data = init(pd.DataFrame(columns))
    # TODO: Make a container for the data so we don't have to load it twice
    original = ECG('output/data.npy.npz', TypeECG.ORIGINAL)
    #reconstructed = ECG('output/data.npy.npz', TypeECG.RECONSTRUCTED)
    #ssert original.__eq__(reconstructed), "Original and reconstructed signals do not match."

    #ecg_number = parameters['start_ecg']
    #lead_number = parameters['start_lead']
    #original.find_r_peaks(ecg_data, parameters['ecg_start'], parameters['ecg_end'], parameters['lead_start'], parameters['lead_end'])
    #original.find_ecg_peaks(ecg_data, parameters['ecg_start'], parameters['ecg_end'], parameters['lead_start'], parameters['lead_end'], parameters['use_plotting'], parameters['print_peaks'], use_show=True)

    #reconstructed.find_r_peaks(ecg_data, ecg_number, TOTAL_NUM_ECGS, lead_number, TOTAL_NUM_LEADS)
    #reconstructed.find_ecg_peaks(ecg_data, ecg_number, TOTAL_NUM_ECGS, lead_number, TOTAL_NUM_LEADS, use_plotting=True, print_peaks=False, use_show=True)
    #save_dataframe(ecg_data, ecg_number, TOTAL_NUM_ECGS, lead_number, TOTAL_NUM_LEADS)
    #order_csv_dataframe(ecg_number, TOTAL_NUM_ECGS, lead_number, TOTAL_NUM_LEADS)