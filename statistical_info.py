import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sktime.forecasting.model_selection import temporal_train_test_split

from generate_dataset import load_btc

df = pd.read_csv(r'D:\semA\COMP 5567\5567project\dataset\interpolate_dataset.csv')
X = df[['avg_price', 'active_addresses', 'google_trends', 'top100_coins_percent', 'avg_polarity']]
y = df['avg_price']


def get_statistics_data(df: DataFrame):
    df1 = df['avg_price'].diff().tolist()
    df_positive = [data for data in df1 if data > 0]
    pos_median = sorted(df_positive)[len(df_positive) // 2]
    pos_mean = sum(df_positive) / len(df_positive)
    df_negative = [data for data in df1 if data < 0]
    neg_median = sorted(df_negative)[len(df_negative) // 2]
    neg_mean = sum(df_negative) / len(df_negative)
    df_neutral = [data for data in df1 if data == 0]
    pos_num = len(df_positive)
    neg_num = len(df_negative)
    return pos_mean, pos_median, neg_mean, neg_median,pos_num,neg_num


df_train, df_test = df[:1710], df[1710:]
pos_mean, pos_median, neg_mean, neg_median, pos_num, neg_num = get_statistics_data(df_train)
print(f"pos mean:{pos_mean}\npos median:{pos_median}\nneg mean:{neg_mean}\nneg median:{neg_median}\npos num:{pos_num}\nneg num:{neg_num}")
pos_mean, pos_median, neg_mean, neg_median, pos_num, neg_num = get_statistics_data(df_test)
print(f"pos mean:{pos_mean}\npos median:{pos_median}\nneg mean:{neg_mean}\nneg median:{neg_median}\npos num:{pos_num}\nneg num:{neg_num}")
print()
