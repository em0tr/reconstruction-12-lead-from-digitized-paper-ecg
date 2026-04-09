import os
import numpy as np
import pandas as pd
import sqlite3
from sklearn.impute import SimpleImputer
from ecg import TypeECG
import util


DB_FILE = 'ecg_data.db'
TABLE_NAME = 'ecg_data'


def save_to_db(df: pd.DataFrame, ecg_type: TypeECG) -> None:
    """
    Save dataframe to database.
    :param df:
    :param ecg_type:
    :return:
    """
    db_name = 'org_' + DB_FILE if ecg_type == ecg_type.ORIGINAL else 'rec_' + DB_FILE
    with sqlite3.connect(create_output_dir() + db_name) as conn:
        df.to_sql('ecg_data', conn, if_exists='replace', index=True)
        conn.commit()



def get_ecg_data_column(ecg_type: TypeECG, column: str) -> pd.DataFrame:
    if ecg_type == ecg_type.ORIGINAL:
        db = 'org_' + DB_FILE
    else:
        db = 'rec_' + DB_FILE
    with sqlite3.connect(create_output_dir() + db) as conn:
        query = f'SELECT {column} FROM {TABLE_NAME}'
        df = pd.read_sql(query, conn)
    return df


def get_ecg_data(ecg_type) -> pd.DataFrame:
    """
    Get the total ECG data as a dataframe.
    :return:
    """
    if ecg_type == ecg_type.ORIGINAL:
        db = 'org_' + DB_FILE
    else:
        db = 'rec_' + DB_FILE
    with sqlite3.connect(create_output_dir() + db) as conn:
        query = f'SELECT * FROM {TABLE_NAME}'
        df = pd.read_sql(query, conn)
    return df


def get_ecg_data_split_org_rec() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the ECG data in two dataframes, one for original and one for reconstructed data.
    :return:
    """
    with sqlite3.connect(create_output_dir() + DB_FILE) as conn:
        org_query = f'SELECT * FROM {TABLE_NAME} WHERE type LIKE "%Original%"'
        org_df = pd.read_sql(org_query, conn)
        rec_query = f'SELECT * FROM {TABLE_NAME} WHERE type LIKE "%Reconstructed%"'
        rec_df = pd.read_sql(rec_query, conn)
    return org_df, rec_df


def get_unsuccessful_ecgs() -> pd.DataFrame:
    with sqlite3.connect(create_output_dir() + DB_FILE) as conn:
        query = f'SELECT * FROM {TABLE_NAME} WHERE success = 0.0'
        df = pd.read_sql(query, conn)
    return df


def create_output_dir() -> str:
    output_dir = 'output/'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_dataframe_csv(df: pd.DataFrame, params: dict) -> None:
    """
    Save the dataframe to csv file.
    :param df: The dataframe.
    :param params: The parameters' dictionary.
    :return: None
    """
    ecg_mean_data_dir = 'csv_ecg_mean_data/'
    os.makedirs(create_output_dir() + ecg_mean_data_dir, exist_ok=True)
    output_name = f'ecg_mean_data_ecg_{params['ecg_start']}_{params['ecg_end']}' \
                  f'_lead_{params['lead_start']}_{params['lead_end']}.csv'
    df.to_csv(create_output_dir() + output_name, index=True)


def order_dataframe(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Reorder the dataframe so that it alternates between original and reconstructed ECGs.
    :param df: The dataframe.
    :param params: The parameters' dictionary.
    :return:
    """
    len_org = (params['ecg_end'] - params['ecg_start']) * (params['lead_end'] - params['lead_start'])
    org_df = df[:len_org:]
    rec_df = df[len_org:]
    org_df['key'] = np.arange(len(org_df)) * 2
    rec_df['key'] = np.arange(len(rec_df)) * 2 + 1
    sorted_df = pd.concat([org_df, rec_df]).sort_values(by=['key']).drop(columns=['key'])
    sorted_df = sorted_df.reset_index(drop=False)
    return sorted_df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    pass


def handle_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some columns end up with NaN values, in this case they are replaced with the mean of the column
    :param df:
    :return:
    """
    list_of_nan_columns = util.feature_columns()
    for column in list_of_nan_columns:
        print(f'Column {column} has {df[column].isna().sum()} NaN out of {len(df[column])} values')
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        df[column] = imputer.fit_transform(df[[column]])

    return df