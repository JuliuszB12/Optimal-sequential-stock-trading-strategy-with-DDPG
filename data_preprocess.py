import time
import datetime
import pandas as pd
import numpy as np
from stockstats import wrap

# Lista aktualnego składu S&P500 z Wikipedii
tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers['Date added'] = pd.to_datetime(tickers['Date added'])
tickers = tickers[tickers['Date added'] < '2014-07-01']
tickers = tickers['Symbol'].tolist()

for i in ['BRK.B', 'BF.B']: # oraz 'BKNG' dla 2e3 max_stock w environment.py
    tickers.remove(i)

# Dane treningowe

start = True
df_500 = None

for ticker in tickers:
    period1 = int(time.mktime(datetime.datetime(2014, 7, 1, 23, 59).timetuple()))
    period2 = int(time.mktime(datetime.datetime(2018, 12, 31, 23, 59).timetuple()))
    interval = '1d'

    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    print(ticker)
    df = pd.read_csv(query_string)
    df = df.drop(['Adj Close'], axis=1)
    df = wrap(df)
    df['tic'] = ticker
    df = df[['tic', 'open', 'high', 'low', 'close', 'volume', 'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'aroon']]
    df = df.reset_index()
    df = df[-df['date'].str.contains("2014")].reset_index()
    if start:
        df_500 = df
        start = False
    else:
        df_500 = pd.concat([df_500, df])

df_500 = df_500.sort_values(by=['date', 'tic'])
df_500.to_pickle('train.pkl')


# Dane testowe

start = True
df_500 = None

for ticker in tickers:
    period1 = int(time.mktime(datetime.datetime(2018, 10, 1, 23, 59).timetuple()))
    period2 = int(time.mktime(datetime.datetime(2019, 4, 30, 23, 59).timetuple()))
    interval = '1d'

    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    print(ticker)
    df = pd.read_csv(query_string)
    df = df.drop(['Adj Close'], axis=1)
    df = wrap(df)
    df['tic'] = ticker
    df = df[['tic', 'open', 'high', 'low', 'close', 'volume', 'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'aroon']]
    df = df.reset_index()
    df = df[pd.to_datetime(df['date']) > '2018-12-31'].reset_index()
    if start:
        df_500 = df
        start = False
    else:
        df_500 = pd.concat([df_500, df])

df_500 = df_500.sort_values(by=['date', 'tic'])
df_500.to_pickle('test.pkl')