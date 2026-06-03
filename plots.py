import json
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from metrics import pearson_correlation
from ecg import LEAD_LABELS, TypeECG
import os
from data import get_ecg_data, get_ecg_data_column
import util
import seaborn as sns


def valid_filetypes(filetype: str) -> bool:
    """
    Valid file types for saving plots.
    :param filetype:
    :return: True if the file type is valid, otherwise False
    """
    filetypes = ['.pdf', '.jpg', '.png']
    if filetype in filetypes:
        return True
    return False


def check_boundaries(df: pd.DataFrame, ecg_num: int, lead_num: int) -> None:
    """
    Ensure that we stay within the boundaries of the ECG signal, i.e. not using lead 13 if there's
    only 12 leads.
    :param df: The dataframe containing the ECG data.
    :param ecg_num: The ECG number to check.
    :param lead_num: The lead number to check.
    :return: None.
    """
    if ecg_num < 0 or ecg_num > df['ecg'].iloc[-1] + 1:
        raise IndexError(f'ECG number {ecg_num} out of range')
    if lead_num < 0 or lead_num > 12:
        raise IndexError(f'Lead number {lead_num} out of range')


def get_ecg_signal(df: pd.DataFrame, ecg_type: TypeECG, ecg_num: int, lead: int) -> np.ndarray:
    """
    Get the ECG signal from a dataframe.
    :param df: The dataframe with the signals.
    :param ecg_type: The type of ECG signal.
    :param ecg_num: The ECG signal number.
    :param lead: The lead.
    :return: A numpy array of the ECG signal.
    """
    check_boundaries(df, ecg_num, lead)
    row = df[
        (df['type'] == ecg_type.name[0] + ecg_type.name[1:].lower()) &
        (df['ecg'] == ecg_num) &
        (df['lead'] == LEAD_LABELS[lead])
    ].iloc[0]
    return np.array(json.loads(row['ecg_signal']))


def create_plot_dir(inner_dir: str) -> str:
    """
    Create a directory for plotting.
    :param inner_dir: Used to create an inner directory in the plots directory to separate different plots.
    :return: A directory path for saving plots.
    """
    plot_dir = 'plots/'
    in_dir = util.INNER_OUTPUT_DIR
    inner = len(util.INNER_OUTPUT_DIR) > 0
    if inner:
        if util.INNER_OUTPUT_DIR[-1] != '/':
            in_dir = util.INNER_OUTPUT_DIR + '/'
    plot_dir += in_dir
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(plot_dir + inner_dir, exist_ok=True)
    return plot_dir + inner_dir


def overlay_ecg_signals(params, final_ecg=100, rows_sub=4, columns_sub=3, fs=500, file_type: str='.pdf') -> None:
    """
    Plot the original and reconstructed ECG signals in the same plot.
    :param params: The parameters' dictionary.
    :param final_ecg: The final ECG signal to plot.
    :param rows_sub: When using subplots specify the rows.
    :param columns_sub: When using subplots specify the columns.
    :param fs: The sampling rate.
    :param file_type: The file type to save if saving plots.
    :return: None.
    """
    org = get_ecg_data(TypeECG.ORIGINAL)
    rec = get_ecg_data(TypeECG.RECONSTRUCTED)
    last_ecg = org['ecg'].iloc[-1] + 1 # Get the index of the last ECG in the dataframe
    if params['use_subplots']:
        for ecg in range(params['ecg_start'], params['ecg_end']):
            if ecg == last_ecg:
                break
            rows = rows_sub
            columns = columns_sub
            fig, axes = plt.subplots(rows, columns, figsize=(20, 10))
            fig.suptitle(f'Original and reconstructed overlay showing all leads for ECG {ecg}', fontsize=16)
            lead = 0
            for i in range(rows):
                for j in range(columns):
                    ax = axes[i, j]
                    ax.set_title(LEAD_LABELS[lead])
                    ax.plot(get_ecg_signal(rec, TypeECG.RECONSTRUCTED, ecg, lead), label="Reconstructed")
                    ax.plot(get_ecg_signal(org, TypeECG.ORIGINAL, ecg, lead), label="Original")
                    ax.grid(True, alpha=0.3)
                    lead += 1
            plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96)) # Adjust layout to prevent overlapping
            plt.legend()
            if params['save_plots']:
                o_dir = 'overlay_signal/'
                plot_name = f'Overlay_ecg_{ecg}_subplots'
                if valid_filetypes(file_type):
                    plt.savefig(create_plot_dir(o_dir) + plot_name + file_type)
                else:
                    print('Invalid filetype for plot, defaulting to .pdf')
                    plt.savefig(create_plot_dir(o_dir) + plot_name + '.pdf')
            if params['ecg_end'] - params['ecg_start'] < 20:
                plt.show()
            plt.close()
    else:
        for ecg in range(params['ecg_start'], params['ecg_end']):
            for lead in range(params['ecg_start'], params['lead_end']):
                o = get_ecg_signal(org, TypeECG.ORIGINAL, ecg, lead)
                r = get_ecg_signal(rec, TypeECG.RECONSTRUCTED, ecg, lead)
                t = np.arange(len(o)) / fs
                plt.figure(figsize=(12, 8))
                plt.plot(t, o, label="Original")
                plt.plot(t, r, label="Reconstructed")
                plt.title(f'Original and reconstructed data signal overlaid for ECG '
                          f'{ecg} {LEAD_LABELS[lead]}', fontsize=20)
                plt.xlabel("Time (sec)", fontsize=16)
                plt.ylabel("Amplitude", fontsize=16)
                plt.tick_params(axis='both', labelsize=18)
                #sns.move_legend(plt, "upper right", fontsize=16, title_fontsize=20)
                plt.legend()
                plt.tight_layout()
                plt.grid(True, alpha=0.3)
                if params['save_plots']:
                    o_dir = 'overlay_signal/'
                    plot_name = f'Overlay_ecg_{ecg}_{LEAD_LABELS[lead]}_plots'
                    if valid_filetypes(file_type):
                        plt.savefig(create_plot_dir(o_dir) + plot_name + file_type)
                    else:
                        print('Invalid filetype for plot, defaulting to .pdf')
                        plt.savefig(create_plot_dir(o_dir) + plot_name + '.pdf')
                plt.show()
                plt.close()
                if ecg == last_ecg:
                    return


def feature_histogram(save_plots: bool=False) -> None:
    """
    Plot the features as histograms.
    :param save_plots: Use to save the plots.
    :return: None
    """
    features = util.feature_columns()
    for feature in features:
        plt.figure(figsize=(12, 12))
        org = get_ecg_data_column(TypeECG.ORIGINAL, feature)
        rec = get_ecg_data_column(TypeECG.RECONSTRUCTED, feature)
        df_columns = ['Original', 'Reconstructed']
        df = pd.concat([org, rec], axis=1)
        df.columns = df_columns
        bins = int(np.sqrt(len(df)))
        ax = sns.histplot(data=df, bins=bins)
        sns.move_legend(ax, "upper right", fontsize=16, title_fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        plt.title(util.df_column_name(feature), fontsize=25)
        plt.xlabel('Values', fontsize=20)
        plt.ylabel('Frequency', fontsize=20)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if save_plots:
            o_dir = 'feature_histograms/'
            plot_name = f'feature_histogram_{feature}.pdf'
            plt.savefig(create_plot_dir(o_dir) + plot_name)
        plt.show()
        plt.close()


def feature_difference(
        save_plots: bool=False,
        plot_type: str='hexbin',
        min_points: bool=False
) -> None:
    """
    Plot the feature differences between original and reconstructed signals.
    :param save_plots: Save the plots
    :param plot_type: Type of plot to show
    :param min_points: Use to reduce the amount of points to show.
    :return: None
    """
    features = util.feature_columns()
    for feature in features:
        plt.figure(figsize=(12, 12))
        org = get_ecg_data_column(TypeECG.ORIGINAL, feature)
        rec = get_ecg_data_column(TypeECG.RECONSTRUCTED, feature)
        if plot_type == 'hexbin':
            plt.hexbin(org, rec, gridsize=50, cmap='plasma', mincnt=5)
            plt.colorbar(label='Count')
            plot_prefix = 'hexbin_'
        else:
            if min_points:
                org = minimize_number_of_points(column=org, method='slice', slice_length=50)
                rec = minimize_number_of_points(column=rec, method='slice', slice_length=50)
            plt.scatter(org, rec, edgecolors='black')
            plt.plot([org.min(), org.max()], [rec.min(), rec.max()], 'r--')
            plot_prefix = 'scatter_'
        plt.title(f'{util.df_column_name(feature)}: Original vs Reconstructed')
        plt.xlabel('Original')
        plt.ylabel('Reconstructed')
        plt.grid(alpha=0.2)
        plt.tight_layout()
        if save_plots:
            o_dir = 'feature_difference/'
            plot_name = plot_prefix + f'feature_difference_{feature}.pdf'
            plt.savefig(create_plot_dir(o_dir) + plot_name)
        plt.show()
        plt.close()


def minimize_number_of_points(column: pd.DataFrame, method: str='slice', slice_length: int=100) -> pd.DataFrame:
    """
    If there are too many points in the columns plots become useless, so this function can be used to
    reduce the number of points.
    :param column: The column to minimize.
    :param method: The method to use for minimizing the number of points.
    :param slice_length: How many points to slice from the column.
    :return: A minimized column or the original column.
    """
    if method == 'slice':
        return column[::slice_length]
    elif method == 'downsample':
        return np.random.choice(len(column), size=1000, replace=False)
    print(f'Invalid minimization method: {method}, returning original column')
    return column


def mu_character() -> str:
    """
    Write the μ character in plots. Use the LaTeX character since it looks better than the copied and
    pasted character.
    :return: The μ character.
    """
    return r"$\mu$"


def bland_altman_plot(
        save_plots: bool=False,
        minimize_points: bool=False,
        file_type: str='.pdf',
) -> None:
    """
    Create bland altman plots of the features.
    :param save_plots: Save the plots.
    :param minimize_points: Minimize the number of points in the plots.
    :param file_type: If saving the plots, select file format.
    :return: None.
    """
    features = util.feature_columns()
    # Process features in groups of 2
    for i in range(0, len(features), 2):
        pair = features[i:i+2]
        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        for ax, feature in zip(axes, pair):
            org = get_ecg_data_column(TypeECG.ORIGINAL, feature)
            rec = get_ecg_data_column(TypeECG.RECONSTRUCTED, feature)
            if minimize_points:
                org = minimize_number_of_points(column=org, method='slice', slice_length=20)
                rec = minimize_number_of_points(column=rec, method='slice', slice_length=20)
            org = np.asarray(org[feature])
            rec = np.asarray(rec[feature])
            mean = np.mean([org, rec], axis=0)
            diff = rec - org
            r_diff, _ = pearson_correlation(mean, diff, feature, sig_diff=False)
            sm.graphics.mean_diff_plot(org, diff, ax=ax)
            ax.set_title(f'{util.df_column_name(feature)}', fontsize=25)
            ax.set_xlabel(f'Real data ({mu_character()}V)', fontsize=20)
            ax.set_ylabel('Reconstruction error', fontsize=20)
            ax.grid(alpha=0.2)
            ax.tick_params(axis='both', labelsize=18)
            ax.text(0.05, 0.05, f'$R^2$ = {(r_diff ** 2):.2f}',
                    transform=ax.transAxes, fontsize=20)
        plt.tight_layout()
        if save_plots:
            o_dir = 'bland_altman_plots/'
            plot_name = f'bland_altman_pair_{i//2 + 1}'
            if file_type[0] != '.':
                file_type = '.' + file_type
            if valid_filetypes(file_type):
                plt.savefig(create_plot_dir(o_dir) + plot_name + file_type)
            else:
                plt.savefig(create_plot_dir(o_dir) + plot_name + '.pdf')

        plt.show()
        plt.close()
