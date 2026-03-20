import numpy as np
import os
import pandas as pd
import argparse
import warnings
from ecg import ECG, TOTAL_NUM_ECGS, TOTAL_NUM_LEADS, TypeECG

columns = {
    'type': [],
    'ecg': [],
    'lead': [],
    'success': [],  # Some ECGs may not work correctly and cause issues
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
