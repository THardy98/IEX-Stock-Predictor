import pandas as pd
import datetime as dt
import os
import pandas_datareader.data as web
import bs4 as bs
import pickle
import requests
from collections import Counter
import numpy as np
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

def sp500_ticker_list():
    source = requests.get('https://www.suredividend.com/sp-500-stocks/')
    soup = bs.BeautifulSoup(source.text, 'lxml')
    table = soup.table
    tickers = []
    for table_row in table.findAll('tr')[1:-1]:
        ticker = table_row.findAll('td')[0]
        tickers.append(ticker.text)

    pickle_tickers = open("sp500tickers.pickle", "wb")
    pickle.dump(tickers, pickle_tickers)
    pickle_tickers.close()
    return tickers

def sp500_data(reload_tickers = False):
    if reload_tickers:
        tickers = sp500_ticker_list()
    else:
        with open('sp500tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2013,1,1)
    end = dt.date.today()

    for ticker in tickers:
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker,"iex", start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('DataFrame and CSV file have already been made for {}'.format(ticker))

def compile_data():
    with open("sp500tickers.pickle","rb") as f:
        tickers = pickle.load(f)

    close_df = pd.DataFrame()

    for ticker in tickers:
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('date', inplace = True)

        df.rename(columns = {'close' : ticker}, inplace = True)
        df.drop(['open','high','low','volume'] , 1, inplace = True)

        if close_df.empty:
            close_df = df
        else:
            close_df = close_df.join(df, how='outer')

    #print(close_df.shape)
    close_df.dropna(thresh = close_df.shape[1]//2 + 1 , inplace = True)
    close_df.dropna(axis = 1, thresh = close_df.shape[0]//2 + 1, inplace = True)
    #print(close_df.head())
    close_df.to_csv('sp500close.csv')
    #print(close_df.shape)

def correlate_data():
    df = pd.read_csv("sp500close.csv")
    df_corr = df.corr()
    df_corr.to_csv("sp500close_corr.csv")

def future_close_for_target(ticker):
    period_day = 7
    df = pd.read_csv('sp500close.csv', index_col=0)
    tickers_vals = df.columns.values.tolist()
    df.fillna(0, inplace = True)

    for i in range(1, period_day + 1):
        df['{}_{}days_close'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    return tickers_vals, df

def buy_sell_hold(*args):
    cols = [c for c in args]
    threshold = 0.02
    for col in cols:
        if col > threshold:
            return 1 #buy
        elif col < -threshold:
            return -1 #sell
    return 0 #hold

def extract_feature_data(ticker):
    tickers_vals, df = future_close_for_target(ticker)

    df['{}_command_vals'.format(ticker)] = list(map(buy_sell_hold,
                                                df['{}_1days_close'.format(ticker)],
                                                df['{}_2days_close'.format(ticker)],
                                                df['{}_3days_close'.format(ticker)],
                                                df['{}_4days_close'.format(ticker)],
                                                df['{}_5days_close'.format(ticker)],
                                                df['{}_6days_close'.format(ticker)],
                                                df['{}_7days_close'.format(ticker)]))


    vals = df['{}_command_vals'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]

    df.fillna(0, inplace = True)

    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace = True)

    df_vals = df[[ticker for ticker in tickers_vals]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_command_vals'.format(ticker)].values

    return X, y, df

def do_ml(ticker):
    X, y, df = extract_feature_data(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print("Accuracy (Confidence): ", confidence)

    predictions = clf.predict(X_test)
    print('Predictions spread:', Counter(predictions))

    return confidence

do_ml("BAC")
