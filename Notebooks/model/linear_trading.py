import numpy as np
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

class Position(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


class LinearSignal:
    def __init__(self, pair, data_obj, window = 200):
        self.pair = pair
        self.window = window
        self.__get_data(data_obj)
        self.__pre_calculation()

    def __get_data(self, data_obj):
        self.sample = data_obj.sample
        self.train_period = data_obj.train_period
        self.test_period = data_obj.test_period
        
        train = self.sample.loc[self.train_period[0]: self.train_period[1]].values
        self.x_train = train[:,0]
        self.y_train = train[:,1]
        test = self.sample.loc[self.test_period[0]: self.test_period[1]].values
        self.x_test = test[:,0]
        self.y_test = test[:,1]
        self.test_length = len(self.x_test)
        pass

    def __pre_calculation(self):
        def beta(df):
            x, y = df[:,0], df[:,1]
            return y.T @ x / (y.T @ y)
        
        rolling_beta = self.sample.rolling(self.window, min_periods = 1, method = 'table')\
            .apply(beta, raw = True, engine= 'numba').fillna(0).values[:,0]
        rolling_spread = self.sample.iloc[:,0] - rolling_beta * self.sample.iloc[:,1]
        self.sigma = rolling_spread.rolling(self.window).std().values[-self.test_length:]
        self.mean = rolling_spread.rolling(self.window).mean().values[-self.test_length:]
        self.spread = rolling_spread.values[-self.test_length:]
        self.beta = rolling_beta[-self.test_length:]
        pass
        
    def get_signals(self):
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

    def calc_total_position(self, ind):
        return self.x_test[ind] + self.beta[ind] * self.y_test[ind]
    
    def calc_pnl_by_trade(self, ind, prev_beta):
        return self.x_test[ind] - prev_beta * self.y_test[ind]

    def get_pnl(self):
        signals = self.get_signals()
        current = Position.FLAT
        
        total_position = self.x_test + self.y_test * self.beta
        pnl = 0
        current_position = 0
        current_beta = 0
        ret = 1
        for ind, p in enumerate(signals):
            
            # Exit current position
            if (current == Position.LONG and p >=0) \
                or (current == Position.SHORT and p >= 0):
                pnl += current.value * self.calc_pnl_by_trade(ind, current_beta)
                ret *= 1 + pnl/current_position
                current = Position.FLAT
            # Enter new position
            if p == 2 and current != Position.SHORT:
                current = Position.SHORT
                pnl = self.spread[ind] # get cash from short
                current_position = self.calc_total_position(ind)
                current_beta = self.beta[ind]
            elif p == -2 and current != Position.LONG:
                current = Position.LONG
                pnl = -self.spread[ind] # pay cash 
                current_position = total_position[ind]
                current_beta = self.beta[ind]
        
        #Exit last position
        if current != Position.FLAT:
            pnl += current.value * self.calc_pnl_by_trade(ind, current_beta)
            ret *= 1 + pnl/current_position
        return ret - 1
    
    def get_cumm_pnl(self):
        signals = self.get_signals()
        current = Position.FLAT
        mkt_values = np.zeros(len(signals))
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
            mkt_values[ind] = ret * (1 + (pnl + current.value * self.spread[ind])/current_position)
        #Exit last position
        if current != Position.FLAT:
            pnl += current.value * self.spread[ind]
            ret *= 1 + pnl/current_position
        mkt_values[ind] = ret * (1 + (pnl + current.value * self.spread[ind])/current_position)
        return mkt_values

    def get_plot(self):
        def beta(df):
            x, y = df[:,0], df[:,1]
            return y.T @ x / (y.T @ y)
        rolling_beta = self.sample.rolling(self.window, min_periods = 1, method = 'table')\
            .apply(beta, raw = True, engine= 'numba').fillna(0).values[:,0]
        spread = self.sample.iloc[:,0] - rolling_beta * self.sample.iloc[:,1]
        s = spread.rolling(self.window).std()
        m = spread.rolling(self.window).mean()
        fig = plt.figure(figsize=(15,5))
        ax = spread.plot()
        ax.plot(m, color='r')
        ax.plot(m+2*s, color='r', linestyle='--')
        ax.plot(m+s, color='r', linestyle='--')
        ax.plot(m-s, color='r', linestyle='--')
        ax.plot(m-2*s, color='r', linestyle='--')
        ax.axvline(self.sample.index[-self.test_length], color='k')
        return fig


class LinearMulti:
    def __init__(self, data_batch):
        self.data_batch = data_batch
    
    def get_pnls(self, list = False):
        if not list:
            pnls = 0
            for data in self.data_batch.values():
                signal = LinearSignal(data.pair, data)
                pnls += signal.get_pnl()
            return pnls
        else:
            pnls = {}
            for data in self.data_batch.values():
                signal = LinearSignal(data.pair, data)
                pnls[data.pair] = signal.get_pnl()
            return pnls


