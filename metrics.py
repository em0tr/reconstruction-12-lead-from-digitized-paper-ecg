import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Union
import util
from data import get_ecg_data_column
from ecg import TypeECG


def basic_stats(ecg: TypeECG, round_num: int=4) -> None:
    features = util.feature_columns()
    name = 'Original' if ecg == TypeECG.ORIGINAL else 'Reconstructed'
    for feature in features:
        column = get_ecg_data_column(ecg, feature)
        print(f'Statistics for {name} {util.df_column_name(feature)}:')
        print(f'Count: {int(column.count().to_numpy()[0])}')
        print(f'Mean: {np.round(column.mean().to_numpy()[0], round_num)}')
        print(f'Mode: {np.round(column.mode().to_numpy(), round_num)}')
        print(f'Median: {np.round(column.median().to_numpy()[0], round_num)}')
        print(f'Variance: {np.round(column.var().to_numpy()[0], round_num)}')
        print(f'Standard Deviation: {np.round(column.std().to_numpy()[0], round_num)}\n')


def calculate_mse() -> None:
    features = util.feature_columns()
    for feature in features:
        org = get_ecg_data_column(TypeECG.ORIGINAL, feature)
        rec = get_ecg_data_column(TypeECG.RECONSTRUCTED, feature)
        diff = (org - rec) ** 2
        mse = diff.sum() / len(diff)
        print(f'Mean squared error for {util.df_column_name(feature)}: {mse.to_numpy()}')
        rmse = np.sqrt(mse)
        print(f'RMSE for {util.df_column_name(feature)}: {rmse.to_numpy()}')
    print()


def calculate_mae() -> None:
    features = util.feature_columns()
    for feature in features:
        org = get_ecg_data_column(TypeECG.ORIGINAL, feature)
        rec = get_ecg_data_column(TypeECG.RECONSTRUCTED, feature)
        diff = np.abs((org - rec))
        mae = diff.sum() / len(diff)
        print(f'Mean absolute error for {util.df_column_name(feature)}: {mae.to_numpy()}')
    print()


def confidence_interval() -> None:
    features = util.feature_columns()
    for feature in features:
        org = get_ecg_data_column(TypeECG.ORIGINAL, feature)
        rec = get_ecg_data_column(TypeECG.RECONSTRUCTED, feature)
        diff = org - rec
        interval = stats.norm.interval(confidence=0.95, loc=diff.mean(), scale=stats.sem(diff))
        print(f'Confidence interval for {util.df_column_name(feature)}: {interval}')
    print()


def student_t_test(org, rec, column: str, sig_diff: bool=False) -> tuple[float, float]:
    assert column in org.columns, f'Column {column} does not exist'
    t_stat, p_value = stats.ttest_rel(rec[column], org[column])
    if sig_diff:
        check_significant_difference(p_value, column, 'Student T test')

    return t_stat, p_value


def f_test(org, rec, column: str, sig_diff: bool=False) -> tuple[float, float]:
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
    alpha = 0.05
    diff = 'Significant difference' if p < alpha else 'Not significant difference'
    if column is None:
        print(f'{test}: {diff}\n')
    else:
        print(f'{test}: {diff} for {util.df_column_name(column)}\n')


def pearson_correlation(org: Union[pd.Series, np.ndarray],
                        rec: Union[pd.Series, np.ndarray],
                        sig_diff: bool=False
                        ) -> tuple[float, float]:
    corr, p_val = stats.pearsonr(rec, org)
    if sig_diff:
        check_significant_difference(p_val, 'Pearson correlation')

    return corr, p_val
