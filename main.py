import pandas as pd
import argparse
from ecg import ECG, TOTAL_NUM_ECGS, TOTAL_NUM_LEADS, TypeECG, LEAD_LABELS
import data
import plots
import metrics
from util import SAMPLING_RATE

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

def df_init(df: pd.DataFrame) -> pd.DataFrame:
    """
    Initialize the dataframe for the mean data points for the ECGs.
    :param df: The dataframe.
    :return: An initialized dataframe.
    """
    df['ecg'] = df['ecg'].astype(int)  # The ECG column should be integer and not float
    df.set_index(['type', 'ecg', 'lead'], inplace=True)  # Use the type, ECG number and lead number as the index
    return df


def set_parameters(params: dict, args: argparse.Namespace) -> None:
    """
    Set the parameters' dict using the command line arguments.
    :param params: The parameter dictionary.
    :param args: The command line arguments.
    :return: None.
    """
    params['process-ecg'] = args.process_ecg
    params['visualize-ecg'] = args.visualize_ecg
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
    params['use_subplots'] = args.use_subplots
    params['save_plots'] = args.save_plots
    params['split-data'] = args.split_data


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(prog=__name__, description='Reconstruction of 12-lead from digitized paper ECG')
    parser.add_argument('--process-ecg', action='store_true', help='Process ECGs to find peaks')
    parser.add_argument('--visualize-ecg', action='store_true', help='Visualize ECGs')
    parser.add_argument('--stats', action='store_true',
                        help='Perform calculations on ECGs to see statistical similarities and differences')
    parser.add_argument('--split-data', action='store_true')
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
    args = parser.parse_args()
    return args


def process_ecgs(params: dict) -> None:
    """
    Process the ECGs to find R-peaks, other peaks and intervals.
    :param params: The parameter dictionary.
    :return: None.
    """
    ecg_data = df_init(pd.DataFrame(columns))
    ecg_type = TypeECG.RECONSTRUCTED
    ecg_input = ECG(input_data='output/data.npy.npz', data_column='reconstructed', type_ecg=ecg_type)

    ecg_input.find_r_peaks(ecg_data, params, print_exception=False)
    ecg_input.find_ecg_peaks(ecg_data, params)
    df = data.handle_nan(ecg_data, ecg_type=ecg_type)
    if params['save_db']:
        data.save_to_db(df, ecg_type=ecg_type)
    if params['save_csv']:
        data.save_dataframe_csv(df, params)


def visualize_ecg(params: dict,
                  overlay: bool=False,
                  difference: bool=False,
                  histogram: bool=False,
                  bland_altman: bool=False
) -> None:
    """
    Visualize ECGs.
    :param params: The parameter dictionary.
    :param overlay: Plot the original and reconstructed ECG signals overlaid.
    :param difference: Plot the difference between original and reconstructed ECG signals.
    :param histogram: Plot histograms of the ECG features.
    :param bland_altman: Plot Bland-Altman plots of the ECG features.
    :return:
    """
    if overlay:
        plots.overlay_ecg_signals(params, final_ecg=10, rows_sub=4, columns_sub=3, fs=SAMPLING_RATE, file_type='.png')
    if difference:
        plots.feature_difference(save_plots=params['save_plots'], plot_type='scatter', min_points=True)
    if histogram:
        plots.feature_histogram(save_plots=params['save_plots'])
    if bland_altman:
        plots.bland_altman_plot(save_plots=params['save_plots'], minimize_points=True, file_type='.pdf')


def stats(
        pearson: bool=False,
        basic_stats: bool=False,
        mae: bool=False,
        mse: bool=False,
        confidence_interval: bool=False,
        percentiles: bool=False,
        t_test: bool=False,
        f_test: bool=False
) -> None:
    """
    Calculate statistical similarities.
    :param pearson: Calculate pearson correlations.
    :param basic_stats: Calculate basic statistics.
    :param mae: Calculate mean absolute error.
    :param mse: Calculate mean squared error and root mean squared error.
    :param confidence_interval: Calculate confidence interval.
    :param percentiles: Calculate percentiles.
    :param t_test: Calculate t-test.
    :param f_test: Calculate f-test.
    :return:
    """
    if pearson:
        metrics.pearson_correlation(sig_diff=True)
    if basic_stats:
        metrics.basic_stats(TypeECG.ORIGINAL, split_leads=False)
        metrics.basic_stats(TypeECG.RECONSTRUCTED, split_leads=False)
    if t_test:
        pass
        #metrics.student_t_test(org=org, rec=rec, column='r_peak_mean', sig_diff=True)
    if f_test:
        pass
        #f_test(org=org, rec=rec, column='r_peak_mean', sig_diff=True)
    if mae:
        metrics.calculate_mae(split_leads=False)
    if mse:
        metrics.calculate_mse(split_leads=False, rounding_digits=6)
    if confidence_interval:
        metrics.confidence_interval(confidence=0.95)
    if percentiles:
        metrics.percentiles(use_all_leads=True, rounding_digits=6)



def split_data(ecg_type: TypeECG, num_leads: int=TOTAL_NUM_LEADS) -> None:
    """
    Create CSV files for each individual lead.
    :param ecg_type: The type of ECG to use.
    :param num_leads: The number of leads in the data.
    :return: None.
    """
    for lead in range(num_leads):
        df = data.get_ecg_data(ecg_type=ecg_type, lead=lead)
        prepend = 'original' if ecg_type == TypeECG.ORIGINAL else 'reconstructed'
        title = f'{prepend}_lead_{LEAD_LABELS[lead]}'
        data.save_dataframe_csv(df, parameters, title=title)


if __name__ == '__main__':
    arguments = parse_arguments()
    parameters = {}
    set_parameters(parameters, arguments)

    if parameters['process-ecg']:
        process_ecgs(parameters)

    if parameters['stats']:
        stats(pearson=False, basic_stats=False, mae=True, mse=False,
              confidence_interval=False, percentiles=False, t_test=False, f_test=False)

    if parameters['visualize-ecg']:
        visualize_ecg(parameters, overlay=True, difference=False, histogram=False, bland_altman=True)

    if parameters['split-data']:
        split_data(TypeECG.RECONSTRUCTED, num_leads=TOTAL_NUM_LEADS)
