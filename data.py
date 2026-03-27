import os
import numpy as np
import pandas as pd
import sqlite3


def save_to_db(df):
    conn = sqlite3.connect('output/ecg_data.db')
    df.to_sql('ecg_data', conn, if_exists='replace', index=True)
    conn.commit()
    conn.close()


def get_ecg_data():
    conn = sqlite3.connect('output/ecg_data.db')
    df = pd.read_sql('select * from ecg_data', conn)
    print(df.head())
    conn.close()


def create_output_dir():
    output_dir = 'output/'
    ecg_mean_data_dir = 'csv_ecg_mean_data/'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + ecg_mean_data_dir, exist_ok=True)
    return output_dir + ecg_mean_data_dir


def save_dataframe_csv(df, params):
    """
    Save the dataframe to csv file.
    :param df: The dataframe.
    :param params: The parameters' dictionary.
    :return: None
    """
    output_name = f'ecg_mean_data_ecg_{params['ecg_start']}_{params['ecg_end']}' \
                  f'_lead_{params['lead_start']}_{params['lead_end']}.csv'
    df.to_csv(create_output_dir() + output_name, index=True)


def order_dataframe(df, params):
    """
    Reorder the csv dataframe so that it alternates between original and reconstructed ECGs.
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