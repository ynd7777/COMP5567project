import datetime
import os

import pandas as pd

"""
trans 4 btc-dataset format
"""


def trans_csv_date(file_paths, features):
    final_df = pd.DataFrame()
    for i, path in enumerate(file_paths):
        df = pd.read_csv(path, header=None)
        current_csv_feature = features[i]
        df.columns = ['date', current_csv_feature]
        if i == 0:
            final_df['date'] = df['date']
        df['date'] = pd.to_datetime(df['date'])
        start_date = datetime.datetime.strptime('20170101', '%Y%m%d')
        end_date = datetime.datetime.strptime('20221114', '%Y%m%d')
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        final_df[current_csv_feature] = df[current_csv_feature]

    return final_df


def trans_all_csv_format(export_path='./dataset/bitcoin_dataset.csv'):
    file_paths = get_all_csv_paths()
    features = ['active_addresses', 'average_price', 'google_trends', 'top100_coins_percent']
    all_files_df = trans_csv_date(file_paths, features)
    all_files_df.to_csv(export_path, index=False)


def get_all_csv_paths(dir_path=r"D:\semA\COMP 5567\5567 Project\original-dataset\btc-dataset"):
    file_paths = [os.path.join(path, file_name) for path, dirs, files in os.walk(dir_path) for file_name in files]
    return file_paths


if __name__ == '__main__':
    # trans_all_csv_format()
    df = pd.read_csv('./dataset/bitcoin_dataset.csv')
    print()
