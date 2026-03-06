from ecg import ECG, TypeECG, SAMPLING_RATE, LEAD_LABELS, TOTAL_NUM_ECGS, TOTAL_NUM_LEADS
import numpy as np
import pandas as pd
import os


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

    len_org = ecg_end * lead_end
    org_df = df[:len_org:]
    rec_df = df[len_org:]
    org_df['key'] = np.arange(len(org_df))*2
    rec_df['key'] = np.arange(len(rec_df))*2+1
    sorted_df = pd.concat([org_df, rec_df]).sort_values(by=['key']).drop(columns=['key'])
    sorted_df = sorted_df.reset_index(drop=True)

    sorted_df.to_csv(ecg_data_dir + 'ordered_' + file, index=False)


if __name__ == '__main__':
    ecg_data = pd.DataFrame({'type': [], 'ecg': [], 'lead': [], 'r_peak_mean': [],
                             't_mean': [], 'p_mean': [], 'q_mean': [], 's_mean': []})
    ecg_data = init(ecg_data)
    # TODO: Make a container for the data so we don't have to load it twice
    original = ECG('output/data.npy.npz', TypeECG.ORIGINAL)
    reconstructed = ECG('output/data.npy.npz', TypeECG.RECONSTRUCTED)
    assert original.__eq__(reconstructed), "Original and reconstructed signals do not match."

    ecg_number = 0
    lead_number = 0
    TOTAL_NUM_ECGS = 2  # Don't want to go through all ECGs yet
    """
    original.find_r_peaks(ecg_data, ecg_number, TOTAL_NUM_ECGS, lead_number, TOTAL_NUM_LEADS)
    original.find_ecg_peaks(ecg_data, ecg_number, TOTAL_NUM_ECGS, lead_number, TOTAL_NUM_LEADS, use_plotting=False, print_peaks=False, use_show=False)
    reconstructed.find_r_peaks(ecg_data, ecg_number, TOTAL_NUM_ECGS, lead_number, TOTAL_NUM_LEADS)
    reconstructed.find_ecg_peaks(ecg_data, ecg_number, TOTAL_NUM_ECGS, lead_number, TOTAL_NUM_LEADS, use_plotting=False, print_peaks=False, use_show=False)
    save_dataframe(ecg_data, ecg_number, TOTAL_NUM_ECGS, lead_number, TOTAL_NUM_LEADS)
    """
    order_csv_dataframe(ecg_number, TOTAL_NUM_ECGS, lead_number, TOTAL_NUM_LEADS)
