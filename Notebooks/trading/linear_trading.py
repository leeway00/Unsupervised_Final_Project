import numpy as np
import pandas as pd
from enum import Enum

class Position(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


class DataPreparation:
    def __init__(self, data: pd.DataFrame or np.array, tickers: list or np.array, 
                 train_period: list or np.array, test_period: list or np.array, 
                 dates = None):
        self.tickers = tickers
        self.dates = dates
        self.train_period = train_period
        self.test_period = test_period
        self.x_train, self.y_train, self.x_test, self.y_test = self.split_xy(data)

    def check_data(self, data):
        if data.shape[1] != len(self.tickers):
            raise ValueError('Data and tickers do not match')
        
        if isinstance(data, pd.DataFrame):
            train = data.iloc[self.train_period].values
            test = data.iloc[self.test_period].values
            self.train_dates = data.index[self.train_period]
            self.test_dates = data.index[self.test_period]
        elif isinstance(data, np.ndarray):
            train_period = self.dates.isin(self.train_period)
            test_period = self.dates.isin(self.test_period)
            train = data[train_period]
            test = data[test_period]
            self.train_dates = self.train_period
            self.test_dates = self.test_period
        return train, test

    def split_xy(self, data):
        train, test = self.check_data(data)
        x_train = train[:,0]
        y_train = train[:,1]
        x_test = test[:,0]
        y_test = test[:,1]
        return x_train, y_train, x_test, y_test


class LinearSignal:
    def __init__(self, x_train, y_train, x_test, y_test, tickers, window = 10):
        self.tickers = tickers
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.test_length = len(x_test)
        self.beta = self.get_ratio()
        self.sigma, self.mean = self.pre_calculation()

    def get_ratio(self):
        return self.y_train.T @ self.x_train /(self.y_train.T @ self.y_train)

    def pre_calculation(self):
        self.formation_spread = self.x_train - self.beta * self.y_train
        sigma = np.std(self.formation_spread)
        mean = np.mean(self.formation_spread)
        return sigma, mean
    
    def get_signals(self):
        self.spread = self.x_test - self.beta * self.y_test
        signal_2sig = self.spread > self.mean + 2 * self.sigma
        signal_1sig = (self.spread > self.mean + self.sigma) & ~signal_2sig
        signal_neg2sig = self.spread < self.mean - 2 * self.sigma
        signal_neg1sig = (self.spread < self.mean - self.sigma) & ~signal_neg2sig
        
        signals = np.zeros(len(self.spread))
        signals[signal_1sig] = 1
        signals[signal_neg1sig] = -1
        signals[signal_2sig] = 2
        signals[signal_neg2sig] = -2
        return signals

    def get_pnl(self):
        signals = self.get_signals()
        current = Position.FLAT
        
        total_position = self.x_test + self.y_test * self.beta
        pnl = 0
        current_position = 0
        ret = 1
        for ind, p in enumerate(signals):
            
            # Exit current position
            if (current == Position.LONG and p >=0) \
                or (current == Position.SHORT and p <= 0):
                pnl += current.value * self.spread[ind]
                ret *= 1 + pnl/current_position
                current = Position.FLAT
                
            # Enter new position
            if p == 2 and current != Position.SHORT:
                current = Position.SHORT
                pnl = self.spread[ind] # get cash from short
                current_position = total_position[ind]
            elif p == -2 and current != Position.LONG:
                current = Position.LONG
                pnl = -self.spread[ind] # pay cash
                current_position = total_position[ind]
        
        #Exit last position
        if current != Position.FLAT:
            pnl += current.value * self.spread[ind]
            ret *= 1 + pnl/current_position
        
        return ret - 1


    def get_cumm_pnl(self):
        signals = self.get_signals()
        current = Position.FLAT
        total_position = self.x_test + self.y_test * self.beta
        pnl = 0
        current_position = 1
        mkt_values = np.zeros(len(signals))
        ret = 1
        for ind, p in enumerate(signals):
            # Exit current position
            if (current == Position.LONG and p >=0) \
                or (current == Position.SHORT and p <= 0):
                pnl += current.value * self.spread[ind]
                ret *= 1 + pnl/current_position
                current = Position.FLAT
                pnl, current_position = 0, 1
            # Enter new position
            if p == 2 and current != Position.SHORT:
                current = Position.SHORT
                pnl = self.spread[ind] # get cash from short
                current_position = total_position[ind]
            elif p == -2 and current != Position.LONG:
                current = Position.LONG
                pnl = -self.spread[ind] # pay cash
                current_position = total_position[ind]
            mkt_values[ind] = ret * (1 + (pnl + current.value * self.spread[ind])/current_position)
        return mkt_values


class Batch_Signal:
    def __init__(self, data, pair_list, train_period, test_period):
        self.data = data
        self.pair_list = pair_list
        self.trian_period = train_period
        self.test_period = test_period
    
    def data_preparation(self):
        data_batch = []
        for pair in self.pair_list:
            data_batch.append(DataPreparation(self.data[pair], self.trian_period, self.test_period))
        return data_batch
    
    def get_pnls(self):
        data_batch = self.data_preparation()
        pnls = 0
        for data in data_batch:
            signal = LinearSignal(data.x_train, data.y_train, data.x_test, data.y_test, data.tickers)
            pnls += signal.get_pnl()
        return pnls
    