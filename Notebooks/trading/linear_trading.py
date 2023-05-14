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
                 window = 10, dates = None):
        self.tickers = tickers
        self.dates = dates
        self.train_period = train_period
        self.test_period = test_period
        self.x_train, self.y_train, self.x_test, self.y_test = self.split_xy(data)
        self.window = window

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
        
    def get_positions(self):
        signals = self.get_signals()
        signals_change = np.diff(signals, prepend=0)
        
        positions = np.zeros(len(signals))
        current = Position.FLAT
        for ind, p in enumerate(signals):
            # Exit current position
            if (current == Position.LONG and p >=0) \
                or (current == Position.SHORT and p <= 0):
                current = Position.FLAT
            # Enter new position
            if p == 2 and current != Position.SHORT:
                current = Position.SHORT
            elif p == -2 and current != Position.LONG:
                current = Position.LONG
            # Save updated position
            positions[ind] = current.value
        self.positions = positions
        return positions
            
    def get_pnl(self):
        positions = self.get_positions()
        cash_flow = np.diff(-positions, prepend=0)
        pnl = sum(cash_flow * self.spread)
        return pnl
        
    def get_cumm_pnl(self):
        positions = self.get_positions()
        cash_flow = self.spread * np.diff(-positions, prepend = 0)
        net_position = np.diff(self.spread * positions, prepend = 0)
        cumm_pnl = np.cumsum(net_position + cash_flow)
        return cumm_pnl
        
        
