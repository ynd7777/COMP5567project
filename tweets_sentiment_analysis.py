import datetime
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from textblob import TextBlob


def trans_tweets_csv():
    df = pd.read_csv(r'D:\semA\COMP 5567\5567 Project\original-dataset\tweets-dataset\tweets-btc.csv',
                     usecols=[0, 3], encoding='ansi')
    df['time'] = pd.to_datetime(df['time'])
    start_time = datetime.datetime.strptime('2017-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    end_time = datetime.datetime.strptime('2022-11-14 23:59:59', '%Y-%m-%d %H:%M:%S')
    df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
    df['date'] = df['time'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d'))
    df.to_csv('./dataset/bitcoin-preprocessing_tweets_dataset.csv', encoding='utf8', index=False)


def read_tweets(path='./dataset/bitcoin-tweets_dataset.csv'):
    return pd.read_csv(path)


def re_clean(text):
    text = re.sub('#bitcoin', 'bitcoin', text, re.IGNORECASE)
    text = re.sub('#btc', 'btc', text, re.IGNORECASE)
    text = re.sub('@btc', 'btc', text, re.IGNORECASE)
    text = re.sub('@bitcoin', 'bitcoin', text, re.IGNORECASE)
    # removes all '\n' string
    text = re.sub('\\n', '', text, re.IGNORECASE)
    # removes all hyperlinks
    text = re.sub('^(https:\S+)', '', text, re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text)
    return text


def data_preprocessing():
    df = read_tweets()
    # all upper letters to lower letters
    df['text'] = df['text'].apply(lambda x: x.lower())
    df['cleaned_text'] = df['text'].apply(lambda x: re_clean(x))
    return df


def get_subjectivity(text: str) -> float:
    # subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective
    return TextBlob(text).sentiment.subjectivity


def get_polarity(text: str) -> float:
    # polarity is a float within the range [-1.0, 1.0],, where -1 refers to negative sentiment and +1 refers to
    # positive sentiment.
    return TextBlob(text).sentiment.polarity


def set_polarity_and_subjectivity_to_df(df: DataFrame):
    df['subjectivity'] = df['cleaned_text'].apply(lambda x: get_subjectivity(x))
    df['polarity'] = df['cleaned_text'].apply(lambda x: get_polarity(x))
    return df


def calculate_average_polarity_per_day(df: DataFrame):
    date_list = df['date'].unique().tolist()
    allday_df = pd.DataFrame()
    for date in date_list:
        oneday_df = df[df['date'] == date]
        # oneday_df['oneday_avg_polarity'] = oneday_df['polarity'].mean()
        oneday_df['oneday_avg_polarity'] = oneday_df['polarity'].mean()
        allday_df = pd.concat([allday_df, oneday_df])
    return allday_df


def calculate_all_polarity(df: DataFrame):
    date_list = df['date'].unique().tolist()
    allday_df = pd.DataFrame()
    for date in date_list:
        oneday_df = df[df['date'] == date]
        # oneday_df['oneday_avg_polarity'] = oneday_df['polarity'].mean()
        oneday_df['oneday_avg_polarity'] = oneday_df['polarity'].mean()
        allday_df = pd.concat([allday_df, oneday_df])
    return allday_df


def get_final_tweets_df(allday_df_path='./dataset/prepocessing_tweets_dataset_with_avg_polarity.csv'):
    allday_df = pd.read_csv(allday_df_path, encoding='ansi')
    final_df = allday_df[['date', 'oneday_avg_polarity']].drop_duplicates()
    return final_df


def draw_tweets_polarity_plot(allday_df: DataFrame):
    target_date = datetime.datetime.strptime('20220915', '%Y%m%d')
    allday_df['date'] = pd.to_datetime(allday_df['date'])
    target_df = allday_df[allday_df['date'] == target_date]
    target_polarity = target_df['polarity']
    print(target_polarity.median())
    print(target_polarity.mean())

    diy_xticks = np.arange(-1.0, 1.0, 0.1)
    plt.xticks(diy_xticks)
    plt.hist(target_polarity, bins=20, histtype='stepfilled', alpha=0.75)
    plt.title('The tweets polarity of 2022-09-15')
    plt.xlabel('polarity')
    plt.ylabel('number of polarity')
    # plt.show()
    plt.savefig('./picture/polarity.png', dpi=200)


if __name__ == '__main__':
    '''df = data_preprocessing()
    df = set_polarity_and_subjectivity_to_df(df)
    df.to_csv('./dataset/preprocessing_tweets_dataset.csv')'''

    df = pd.read_csv('./dataset/bitcoin-tweets-dataset.csv')
    draw_tweets_polarity_plot(df)

    '''tweets_df = get_final_tweets_df()
    tweets_df['date'] = pd.to_datetime(tweets_df['date'])
    start_date = datetime.datetime.strptime('20170101', '%Y%m%d')
    end_date = datetime.datetime.strptime('20221110', '%Y%m%d')
    tweets_df = tweets_df[(tweets_df['date'] >= start_date) & (tweets_df['date'] <= end_date)]
    tweets_df.to_csv('./dataset/tweets_perday_polarity.csv')
    btc_df = pd.read_csv('dataset/dataset.csv')
    btc_df['date'] = pd.to_datetime(btc_df['date'])
    tweets_date_list = tweets_df['date'].unique().tolist()
    btc_date_list = btc_df['date'].unique().tolist()
    print(len(tweets_date_list), len(btc_date_list))
    print(tweets_date_list, btc_date_list, sep='\n')
    for date in btc_date_list:
        if date not in tweets_date_list:
            print(date)'''
