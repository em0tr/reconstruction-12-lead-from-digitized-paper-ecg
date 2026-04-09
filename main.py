import pandas as pd
import argparse
from ecg import ECG, TOTAL_NUM_ECGS, TOTAL_NUM_LEADS, TypeECG
import data
import plots
import metrics

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


def set_parameters(params, args):
    """
    Set the parameters' dict using the command line arguments.
    :param params:
    :param args:
    :return:
    """
    params['process_ecg'] = args.process_ecg
    params['visualize_ecg'] = args.visualize_ecg
    params['stats'] = args.stats
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
    params['save_db'] = args.save_db
    params['delineate_method'] = args.delineate_method
    params['process_method'] = args.process_method
    params['quality_method'] = args.quality_method
    params['print_keys'] = args.print_keys
    params['use_subplots'] = args.use_subplots
    params['save_plots'] = args.save_plots


def parse_arguments():
    """
    Parse command line arguments
    :return:
    """
    parser = argparse.ArgumentParser(prog=__name__, description='Reconstruction of 12-lead from digitized paper ECG')
    parser.add_argument('--process_ecg', action='store_true', help='Process ECGs to find peaks')
    parser.add_argument('--visualize_ecg', action='store_true', help='Visualize ECGs')
    parser.add_argument('--stats', action='store_true',
                        help='Perform calculations on ECGs to see statistical similarities and differences')
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
    parser.add_argument('--use_subplots', action='store_true',
                        help='Use subplots to make all leads for an ECG be in a single plot')
    parser.add_argument('--save_plots', action='store_true', help='Save plots')
    parser.add_argument('--use_segment', action='store_true', help='Show ECGs as segments')
    parser.add_argument('--print_mean', action='store_true',
                        help='Print the mean values for peaks and intervals')
    parser.add_argument('--save_csv', action='store_true', help='Save the ECGs to a csv file')
    parser.add_argument('--save_db', action='store_true', help='Save the ECGs to an SQLite file')
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


def process_ecgs(params):
    """
    Process the ECGs to find R-peaks, other peaks and intervals.
    :param params:
    :return:
    """
    ecg_data = init(pd.DataFrame(columns))
    # TODO: Make a container for the data so we don't have to load it twice
    original = ECG('output/data.npy.npz', TypeECG.ORIGINAL)
    #reconstructed = ECG('output/data.npy.npz', TypeECG.RECONSTRUCTED)
    #assert original.__eq__(reconstructed), "Original and reconstructed signals do not match."

    original.find_r_peaks(ecg_data, params)
    original.find_ecg_peaks(ecg_data, params)

    #reconstructed.find_r_peaks(ecg_data, params)
    #reconstructed.find_ecg_peaks(ecg_data, params)

    #with warnings.catch_warnings():
        #warnings.simplefilter("ignore")  # Ignore SettingWithCopyWarning warnings
        #ordered_df = order_dataframe(ecg_data, params)

    df = data.handle_nan(ecg_data)

    if params['save_db']:
        data.save_to_db(df, ecg_type=TypeECG.ORIGINAL)

    if params['save_csv']:
        data.save_dataframe_csv(df, params)


def visualize_ecg(params):
    #plots.overlay_ecg_signals(params)
    plots.feature_difference(save_plots=params['save_plots'], plot_type='scatter', min_points=True)
    #plots.feature_histogram(save_plots=params['save_plots'])
    #plots.bland_altman_plot(org=org, rec=rec, column='r_peak_mean', save_plots=params['save_plots'])
    #plots.bland_altman_plot(org=org, rec=rec, column='t_mean', save_plots=params['save_plots'])
    #plots.bland_altman_plot(org=org, rec=rec, column='p_mean', save_plots=params['save_plots'])
    #plots.bland_altman_plot(org=org, rec=rec, column='q_mean', save_plots=params['save_plots'])
    #plots.bland_altman_plot(org=org, rec=rec, column='s_mean', save_plots=params['save_plots'])
    #plots.bland_altman_plot(org=org, rec=rec, column='rr_interval_mean', save_plots=params['save_plots'])
    #plots.bland_altman_plot(org=org, rec=rec, column='qt_interval_mean', save_plots=params['save_plots'])
    #plots.bland_altman_plot(org=org, rec=rec, column='pr_interval_mean', save_plots=params['save_plots'])


def stats(params):
    #p, _ = pearson_correlation(org=org['r_peak_mean'], rec=rec['r_peak_mean'], sig_diff=True)
    #student_t_test(org=org, rec=rec, column='r_peak_mean', sig_diff=True)
    #f_test(org=org, rec=rec, column='r_peak_mean', sig_diff=True)
    #calculate_mae()
    #calculate_mse()
    #confidence_interval()
    metrics.basic_stats(TypeECG.ORIGINAL)
    metrics.basic_stats(TypeECG.RECONSTRUCTED)


if __name__ == '__main__':
    arguments = parse_arguments()
    parameters = {}
    set_parameters(parameters, arguments)

    if parameters['process_ecg']:
        process_ecgs(parameters)

    if parameters['stats']:
        stats(parameters)

    if parameters['visualize_ecg']:
        visualize_ecg(parameters)
