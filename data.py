import os
import numpy as np
import pandas as pd
import sqlite3
from sklearn.impute import SimpleImputer
from ecg import TypeECG
from util import *


def save_to_db(df: pd.DataFrame, ecg_type: TypeECG, name: str=None) -> None:
    """
    Save dataframe to database.
    :param df: The dataframe to save.
    :param ecg_type: The type of ecg data to save.
    :param name: Specify a name for the db.
    :return: None
    """
    if name is None:
        db_name = 'org_' + DB_FILE if ecg_type == ecg_type.ORIGINAL else 'rec_' + DB_FILE
    else:
        prepend = 'org_' + DB_FILE if ecg_type == ecg_type.ORIGINAL else 'rec_' + DB_FILE
        db_name = prepend + name + '.db'
    with sqlite3.connect(create_output_dir() + db_name) as conn:
        df.to_sql('ecg_data', conn, if_exists='replace', index=True)
        conn.commit()


def get_ecg_data_column(ecg_type: TypeECG, column: str, lead: int=None) -> pd.DataFrame:
    db = 'org_' + DB_FILE if ecg_type == ecg_type.ORIGINAL else 'rec_' + DB_FILE
    with sqlite3.connect(create_output_dir() + db) as conn:
        if lead is not None:
            assert 0 <= lead < TOTAL_NUM_LEADS, f'Lead must be between 0 and {TOTAL_NUM_LEADS}'
            query = f'SELECT {column} FROM {TABLE_NAME} WHERE lead = "{LEAD_LABELS[lead]}"'
        else:
            query = f'SELECT {column} FROM {TABLE_NAME}'
        df = pd.read_sql(query, conn)
    return df


def get_ecg_data(ecg_type: TypeECG, lead: int=None) -> pd.DataFrame:
    """
    Get the total ECG data as a dataframe. Supplying a lead parameter lets you fetch
    ECG data for only that lead.
    :param ecg_type: The type of ecg data to get.
    :param lead: The lead to use.
    :return:
    """
    db = 'org_' + DB_FILE if ecg_type == ecg_type.ORIGINAL else 'rec_' + DB_FILE
    with sqlite3.connect(create_output_dir() + db) as conn:
        if lead is not None:
            assert 0 <= lead < TOTAL_NUM_LEADS, f'Lead must be between 0 and {TOTAL_NUM_LEADS}'
            query = f'SELECT * FROM {TABLE_NAME} WHERE lead = "{LEAD_LABELS[lead]}"'
        else:
            query = f'SELECT * FROM {TABLE_NAME}'
        df = pd.read_sql(query, conn)
    return df


def get_ecg_data_split_org_rec() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the ECG data in two dataframes, one for original and one for reconstructed data.
    This is useful if both the original and reconstructed data is saved in the same db.
    :return:
    """
    with sqlite3.connect(create_output_dir() + DB_FILE) as conn:
        org_query = f'SELECT * FROM {TABLE_NAME} WHERE type LIKE "%Original%"'
        org_df = pd.read_sql(org_query, conn)
        rec_query = f'SELECT * FROM {TABLE_NAME} WHERE type LIKE "%Reconstructed%"'
        rec_df = pd.read_sql(rec_query, conn)
    return org_df, rec_df


def get_unsuccessful_ecgs(ecg_type: TypeECG=None) -> pd.DataFrame:
    """
    Get all unsuccessful ECGs. By not providing a type for the ECG, assume you're after
    a db file with both original and reconstructed data. Otherwise, the ECG type is used
    if the original and reconstructed data is in different db files.
    :param ecg_type:
    :return:
    """
    if ecg_type is None:
        db = DB_FILE
    elif ecg_type == ecg_type.ORIGINAL:
        db = 'org_' + DB_FILE
    else:
        db = 'rec_' + DB_FILE

    assert os.path.isfile(create_output_dir() + db), f'{db} does not exist'
    with sqlite3.connect(create_output_dir() + db) as conn:
        query = f'SELECT * FROM {TABLE_NAME} WHERE success = 0.0'
        df = pd.read_sql(query, conn)
    return df


def create_output_dir() -> str:
    """
    Create the output directory. If the INNER_OUTPUT_DIR is defined it will be created in the output/ directory.
    :return: The output directory.
    """
    output_dir = 'output/'
    os.makedirs(output_dir, exist_ok=True)
    inner = len(INNER_OUTPUT_DIR) > 0
    if inner:
        in_dir = INNER_OUTPUT_DIR
        if in_dir[-1] != '/':
            in_dir += '/'
        output_dir += in_dir
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    return output_dir


def save_dataframe_csv(df: pd.DataFrame, params: dict, title: str=None) -> None:
    """
    Save the dataframe to csv file.
    :param df: The dataframe.
    :param params: The parameters' dictionary.
    :param title: The title of the CSV file.
    :return: None
    """
    if title is not None:
        os.makedirs(create_output_dir(), exist_ok=True)
        output_name = title + '.csv'
    else:
        output_name = f'ecg_mean_data_ecg_{params['ecg_start']}_{params['ecg_end']}' \
                      f'_lead_{params['lead_start']}_{params['lead_end']}.csv'
        os.makedirs(create_output_dir(), exist_ok=True)
    if 'ecg_signal' in df.columns:
        # ECG signal column makes CSV want to die and should only be in SQLite files
        df = df.drop(columns=['ecg_signal'])
    if 'r_peaks' in df.columns:
        # Not considered a feature column so it doesn't need to be saved to CSV
        df = df.drop(columns=['r_peaks'])

    df.to_csv(create_output_dir() + output_name, index=False)


def order_dataframe(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Reorder the dataframe so that it alternates between original and reconstructed ECGs. Can be used
    if the original and reconstructed data is in the same dataframe.
    :param df: The dataframe.
    :param params: The parameters' dictionary.
    :return: An ordered dataframe.
    """
    len_org = (params['ecg_end'] - params['ecg_start']) * (params['lead_end'] - params['lead_start'])
    org_df = df[:len_org:]
    rec_df = df[len_org:]
    org_df['key'] = np.arange(len(org_df)) * 2
    rec_df['key'] = np.arange(len(rec_df)) * 2 + 1
    sorted_df = pd.concat([org_df, rec_df]).sort_values(by=['key']).drop(columns=['key'])
    sorted_df = sorted_df.reset_index(drop=False)
    return sorted_df


def handle_nan(df: pd.DataFrame, ecg_type: TypeECG) -> pd.DataFrame:
    """
    Some columns end up with NaN values, in this case they are replaced with the mean of the column. The
    number of NaNs per column is written to a file before being handled.
    :param df:
    :param ecg_type:
    :return:
    """
    list_of_nan_columns = feature_columns()
    prepend = 'org_' if ecg_type == ecg_type.ORIGINAL else 'rec_'
    # Empty file before writing
    open(create_output_dir() + prepend + 'number_of_nans_in_dataframe' + '.txt', 'w').close()
    for column in list_of_nan_columns:
        with open(create_output_dir() + prepend + 'number_of_nans_in_dataframe' + '.txt', 'a') as file:
            num_nan_percent = round((df[column].isna().sum() / len(df[column])) * 100)
            file.write(f'Column {column} has {df[column].isna().sum()} NaN out of {len(df[column])} values '
                       f'({num_nan_percent}%)\n')
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        df[column] = imputer.fit_transform(df[[column]])
    return df