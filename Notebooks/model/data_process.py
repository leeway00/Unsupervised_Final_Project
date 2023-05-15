import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

class TradingData:
    def __init__(self, train_x, train_y, test_x, test_y, pair):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.pair = pair

class DataPrepation:
    def __init__(self, df_data, df_ticker):
        self.df_data = df_data
        self.df_ticker = df_ticker
        self.scaler = StandardScaler()
        self.tickers = None
        self.price_data = None
    
    def __union_tickers(self, list_of_lists):
        ticker_set = set()
        for i in list_of_lists:
            set2 = set(i.split(","))
            ticker_set = ticker_set.union(set2)
        self.tickers = list(ticker_set)

    # Preprocess the raw data, by cutting the data into given start/end date. 
    # Dates should be string here.
    def __get_all_data(self, start, end):
        tickers = self.df_ticker.loc[start:end].tickers
        self.__union_tickers(tickers)
        
        df = self.df_data.loc[start:end]
        df = df[df.ticker.isin(self.tickers)]
        to_drop = df.loc[df.isna().any(axis=1)]['ticker'].unique()
        df = df[~df.ticker.isin(to_drop)]
        return df

    
    def dimension_reduction_preprocess(self, start, end):
        # Preprocessing steps
        df_period = self.__get_all_data(start, end)
        df_train = df_period.reset_index().sort_values(['ticker', 'date'])
        idx = df_train[['ticker', 'date']].values
        df_train = df_train.drop(['date','ticker'], axis=1)

        ohe_column = 'gicdesc'
        ohe_categories = df_train[ohe_column].unique().tolist()
        enc = OneHotEncoder(sparse_output=False, categories=[ohe_categories]) 
        transformer = make_column_transformer((enc, [ohe_column]), remainder='passthrough') 
        X_train = transformer.fit_transform(df_train)
        
        X_train = self.scaler.fit_transform(X_train)
        return X_train, idx
    
    
    def __get_price_data(self, start, end):
        if not self.tickers:
            tickers_temp = self.df_ticker.loc[start:end].tickers
            self.__union_tickers(tickers_temp)
        global temp
        df = self.df_data[['ticker','close']].loc[start:end]
        
        df = df[df.ticker.isin(self.tickers)]
        df  = df.reset_index().set_index(['date', 'ticker']).stack()
        df.index = df.index.droplevel(2)
        df = df.unstack(1)
        return df
        
    def trading_preprocess(self, pair_list, train_start, train_end, test_start, test_end):
        df = self.__get_price_data(train_start, test_end)
        df_processer = dict()
        for pair in pair_list:
            x, y =  pair
            train_x = df[x].loc[train_start:train_end].values
            train_y = df[y].loc[train_start:train_end].values
            test_x = df[x].loc[test_start:test_end].values
            test_y = df[y].loc[test_start:test_end].values
            df_processer[pair] = TradingData(train_x, train_y,
                                             test_x, test_y, pair)
        return df_processer
