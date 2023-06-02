import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

class TradingData:
    """ 
    This class is used to store the data for trading model
    """
    def __init__(self, pair, sample, train_start, train_end, test_start, test_end):
        self.pair = pair
        self.sample = sample
        self.train_period = (train_start, train_end)
        self.test_period = (test_start, test_end)


class DataPrepation:
    """
    Process the data, for
    1) Dimension reduction : dimension_reduction_preprocess
    2) Trading model : trading_preprocess
    3) Get the price table : price, price_norm
    
    Only above four public functions are actually used
    """
    
    def __init__(self, df_data, df_ticker):
        """__init__ _summary_

        :param df_data: pd.DataFrame of the raw data including price
        :param df_ticker: pd.DataFrame of ticker list by date
        """
        self.df_data = df_data
        self.df_ticker = df_ticker
        self.scaler = StandardScaler()
        self.tickers = None
        self.__price = self.__price_table(df_data)
        
    
    def __union_tickers(self, list_of_lists):
        # Private function to union all the tickers in the list of lists
        ticker_set = set()
        for i in list_of_lists:
            set2 = set(i.split(","))
            ticker_set = ticker_set.union(set2)
        self.tickers = list(ticker_set)

    def __price_table(self, df):
        # Private function to get the price table
        price = pd.pivot_table(df, index = df.index, columns = 'ticker', values = 'close')
        return price
    
    def price(self, start, end):
        # Get the price table between start and end date
        return self.__price.loc[start:end]
    
    def price_norm(self, start, end):
        # Get the normalized price table between start and end date
        # Used to test the stationarity
        price = self.price(start, end)
        return price / price.iloc[0]
    
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
        """dimension_reduction_preprocess 
        preprocess data for dimensionality reduction
        
        :return:
        X_train : scaled and encoded company features 
        idx : ticker and date for each index 
        price_norm : normalized price
        """
        
        # Preprocessing steps
        df_period = self.__get_all_data(start, end)
        price_norm = pd.pivot_table(df_period, index = df_period.index, columns = 'ticker', values = 'close')
        ticker_avail = price_norm.columns[~price_norm.isna().any()]
        price_norm = price_norm[ticker_avail]
        price_norm /= price_norm.iloc[0]
        price_norm = price_norm.unstack().values.reshape(-1,1)
        
        df_period = df_period[df_period.ticker.isin(ticker_avail)]
        df_train = df_period.reset_index().sort_values(['ticker', 'date'])
        idx = df_train[['ticker', 'date']].values
        df_train = df_train.drop(['date','ticker'], axis=1)

        ohe_column = 'gicdesc'
        ohe_categories = df_train[ohe_column].unique().tolist()
        enc = OneHotEncoder(sparse_output=False, categories=[ohe_categories]) 
        transformer = make_column_transformer((enc, [ohe_column]), remainder='passthrough') 
        X_train = transformer.fit_transform(df_train)
        X_train = self.scaler.fit_transform(X_train)
        
        # price_norm = pd.pivot_table(df_period, index = df_period.index, columns = 'ticker', values = 'close')
        # price_norm /= price_norm.iloc[0]
        # price_norm = price_norm.unstack().values.reshape(-1,1)
        return X_train, idx, price_norm
    
    def __get_price_data(self, start, end):
        if not self.tickers:
            tickers_temp = self.df_ticker.loc[start:end].tickers
            self.__union_tickers(tickers_temp)

        df = self.df_data[['ticker','close']].loc[start:end]
        df = pd.pivot_table(df, index = df.index, columns = 'ticker', values = 'close')
        return df
        
    def trading_preprocess(self, pair_list, train_start, train_end, test_start, test_end):
        """trading_preprocess _summary_
        Preprocess the data for trading model.
        Can be directly used for linear_trading.LinearMulti
        
        :return: dictionary of {pair: datat_process.TradingData object}
        """
        
        df = self.__get_price_data(train_start, test_end)
        df_processer = dict()
        for pair in pair_list:
            sample = df[list(pair)].loc[train_start:test_end]
            df_processer[pair] = TradingData(pair, sample, train_start, train_end, test_start, test_end)
        return df_processer
