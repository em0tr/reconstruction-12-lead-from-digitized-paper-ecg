import numpy as np
import scipy.stats as stats
import util
from data import get_ecg_data_column
from ecg import TypeECG
from util import LEAD_LABELS, TOTAL_NUM_LEADS


def basic_stats(ecg: TypeECG, round_num: int=4) -> None:
    """
    Get basic statistics about the ECG data.
    :param ecg: Original or reconstructed ECG data.
    :param round_num: How many digits to round to.
    :return: None
    """
    features = util.feature_columns()
    name = 'Original' if ecg == TypeECG.ORIGINAL else 'Reconstructed'
    for feature in features:
        for lead in range(TOTAL_NUM_LEADS):
            column = get_ecg_data_column(ecg_type=ecg, column=feature, lead=lead)
            print(f'Statistics for {name} {util.df_column_name(feature)} lead {LEAD_LABELS[lead]}:')
            print(f'Count: {int(column.count().to_numpy()[0])}')
            print(f'Mean: {np.round(column.mean().to_numpy()[0], round_num)}')
            print(f'Mode: {np.round(column.mode().to_numpy(), round_num)}')
            print(f'Median: {np.round(column.median().to_numpy()[0], round_num)}')
            print(f'Variance: {np.round(column.var().to_numpy()[0], round_num)}')
            print(f'Standard Deviation: {np.round(column.std().to_numpy()[0], round_num)}\n')


def calculate_mse() -> None:
    """
    Calculate mean squared error (MSE) and root mean squared error (RMSE).
    :return: None
    """
    features = util.feature_columns()
    for feature in features:
        for lead in range(TOTAL_NUM_LEADS):
            org = get_ecg_data_column(TypeECG.ORIGINAL, feature, lead)
            rec = get_ecg_data_column(TypeECG.RECONSTRUCTED, feature, lead)
            diff = (org - rec) ** 2
            mse = diff.sum() / len(diff)
            print(f'Mean squared error for {util.df_column_name(feature)} lead {LEAD_LABELS[lead]}: {mse.to_numpy()[0]}')
            rmse = np.sqrt(mse)
            print(f'RMSE for {util.df_column_name(feature)} lead {LEAD_LABELS[lead]}: {rmse.to_numpy()[0]}')
    print()


def calculate_mae() -> None:
    """
    Calculate mean absolute error (MAE).
    :return: None
    """
    features = util.feature_columns()
    for feature in features:
        org = get_ecg_data_column(TypeECG.ORIGINAL, feature)
        rec = get_ecg_data_column(TypeECG.RECONSTRUCTED, feature)
        diff = np.abs(org - rec)
        mae = diff.sum() / len(diff)
        print(f'Mean absolute error for {util.df_column_name(feature)}: {mae.to_numpy()}')
    print()


def confidence_interval() -> None:
    """
    Calculate confidence interval.
    :return: None
    """
    features = util.feature_columns()
    confidence = 0.95
    for feature in features:
        org = get_ecg_data_column(TypeECG.ORIGINAL, feature)
        rec = get_ecg_data_column(TypeECG.RECONSTRUCTED, feature)
        diff = org - rec
        interval = stats.norm.interval(confidence=confidence, loc=diff.mean(), scale=stats.sem(diff))
        print(f'{int(confidence*100)}% confidence interval for {util.df_column_name(feature)}: {interval}')
    print()


def student_t_test(org, rec, column: str, sig_diff: bool=False) -> tuple[float, float]:
    """
    Calculate Student T-test statistic.
    :param org:
    :param rec:
    :param column:
    :param sig_diff:
    :return: None
    """
    assert column in org.columns, f'Column {column} does not exist'
    t_stat, p_value = stats.ttest_rel(rec[column], org[column])
    if sig_diff:
        check_significant_difference(p_value, column, 'Student T test')

    return t_stat, p_value


def f_test(org, rec, column: str, sig_diff: bool=False) -> tuple[float, float]:
    """
    Calculate F-test statistic.
    :param org:
    :param rec:
    :param column:
    :param sig_diff:
    :return: None
    """
    assert column in org.columns, f'Column {column} does not exist'
    org_var = org[column].var()
    rec_var = rec[column].var()

    if org_var >= rec_var:
        f_stat = org_var / rec_var
        dfn = len(org[column]) - 1
        dfd = len(rec[column]) - 1
    else:
        f_stat = rec_var / org_var
        dfn = len(rec[column]) - 1
        dfd = len(org[column]) - 1
    p_value = stats.f.sf(f_stat, dfn, dfd) * 2
    print(f'F-statistic = {f_stat}')
    print(f'P-value = {p_value}')
    if sig_diff:
        check_significant_difference(p_value, column, 'Student F test')
    return f_stat, p_value


def check_significant_difference(p: float, test: str, column: str=None) -> None:
    """
    Use the p-value from the tests to check the significant difference.
    :param p:
    :param test:
    :param column:
    :return: None
    """
    alpha = 0.05
    diff = 'Significant difference' if p < alpha else 'Not significant difference'
    if column is None:
        print(f'{test}: {diff}')
    else:
        print(f'{test}: {diff} for {util.df_column_name(column)}')


def pearson_correlation(
        org: np.ndarray=None,
        rec: np.ndarray=None,
        column: str=None,
        sig_diff: bool=False
) -> None | tuple[float, float]:
    """
    Calculate Pearson correlation.
    :param org:
    :param rec:
    :param column:
    :param sig_diff:
    :return: None | tuple[float, float]
    """
    if org is not None and rec is not None:
        org = get_ecg_data_column(TypeECG.ORIGINAL, column)
        rec = get_ecg_data_column(TypeECG.RECONSTRUCTED, column)
        corr, p_val = stats.pearsonr(org[column], rec[column])
        if sig_diff:
            check_significant_difference(p_val, column, 'Pearson correlation')
        return corr, p_val
    else:
        features = util.feature_columns()
        for feature in features:
            org = get_ecg_data_column(TypeECG.ORIGINAL, feature)
            rec = get_ecg_data_column(TypeECG.RECONSTRUCTED, feature)
            corr, p_val = stats.pearsonr(rec, org)
            print(f'Pearson correlation: {corr[0]:.6f} for column {util.df_column_name(feature)}')
            if sig_diff:
                check_significant_difference(p_val, 'Pearson correlation', feature)
        print()
        return None
