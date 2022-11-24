import numpy as np
import pandas as pd
from pandas import DataFrame, concat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_features_and_labels(df: DataFrame, timesteps=5):
    """
    sliding window: windows size=5
    :param df:
    :param timesteps:
    :return:
    """
    features_list = []
    labels_list = []
    features = df[['avg_price', 'active_addresses', 'google_trends', 'top100_coins_percent', 'avg_polarity']]
    labels = df['avg_price']
    row = df.shape[0]

    for i in range(0, row - timesteps, 1):
        # print(i,i + timesteps-1,i+timesteps)
        # 1-5
        features_data = features.iloc[i:i + timesteps]
        # 6
        label = labels.iloc[i + timesteps]
        features_list.append(features_data)
        labels_list.append(label)

    return features_list, labels_list


def do_interpolate():
    df = pd.read_csv(r'D:\semA\COMP 5567\5567project\dataset\dataset.csv')
    df.index = pd.to_datetime(df['date'])
    df = df.interpolate(method='time')
    df['active_addresses'] = df['active_addresses'].astype(int)
    # df.to_csv('./dataset/interpolate_dataset.csv',index=False)


def load_btc():
    df = pd.read_csv(r'D:\semA\COMP 5567\5567project\dataset\interpolate_dataset.csv')
    df_rows = int(df.shape[0] * 0.8)
    df_train, df_test = df.iloc[:df_rows], df.iloc[df_rows:]
    X_train, y_train = get_features_and_labels(df_train)
    X_test, y_test = get_features_and_labels(df_test)
    train_rows = np.shape(X_train)[0]
    train_columns = np.shape(X_train)[1] * np.shape(X_train)[2]
    test_rows = np.shape(X_test)[0]
    test_columns = np.shape(X_test)[1] * np.shape(X_test)[2]
    X_train = np.reshape(X_train, (train_rows, train_columns))
    X_test = np.reshape(X_test, (test_rows, test_columns))
    print(X_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_btc()

    '''df = pd.read_csv(r'D:\semA\COMP 5567\5567project\dataset\interpolate_dataset.csv')
    X = df[['avg_price', 'active_addresses', 'google_trends', 'top100_coins_percent', 'avg_polarity']]
    y = df['avg_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    '''
