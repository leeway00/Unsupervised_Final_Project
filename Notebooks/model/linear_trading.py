import numpy as np
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

class Position(Enum):
    """
    Indicating the position during the trade
    """
    LONG = 1
    SHORT = -1
    FLAT = 0


class LinearSignal:
    """
    Creating Backtest Signals based on Linear Assumption
    """
    def __init__(self, pair, data_obj, window = 200):
        self.pair = pair
        self.window = window
        self.__get_data(data_obj)
        self.__pre_calculation()

    def __get_data(self, data_obj):
        """__get_data _summary_
        parse the data object into x_train, y_train, x_test, y_test

        :param data_obj: data_process.TradingData object
        """
        
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
        """__pre_calculation _summary_
        Calculate the rolling beta, spread, mean and sigma on the test period
        Spread: x - beta * y, beta is rolling beta
        """
        def beta(df):
            """beta _summary_
            Calculate the beta of the linear regression
            Assuming that when the spread is stationary without trend,
            we can get the beta by linear regression of x = alpha + beta * y + epsilon
            which will leave x - beta * y = alpha + epsilon as stationary spread
            """
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
        """get_signals _summary_
        Each signals is a number from -2 to 2, indicating the position of the spread
        -2: spread crossed 2 sigma below mean
        -1: spread crossed 1 sigma below mean
        0: spread between 1 sigma below and 1 sigma above mean
        1: spread crossed 1 sigma above mean
        2: spread crossed 2 sigma above mean

        :return: np.array of signals
        """
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
        """calc_total_position _summary_
        Calculate the total position of the pairs trading, 
        on the entrace of the trade
        """
        return self.x_test[ind] + self.beta[ind] * self.y_test[ind]
    
    def calc_pnl_by_trade(self, ind, prev_beta):
        """calc_pnl_by_trade _summary_
        Calculate the pnl of the trade when exit the position
        Given the previous beta (the beta when enter the trade)
        """
        return self.x_test[ind] - prev_beta * self.y_test[ind]

    def get_pnl(self):
        """get_pnl _summary_
        get pnl of the trading strategy

        :return: float pnl
        """
        signals = self.get_signals()
        current = Position.FLAT
        
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
                current_position = self.calc_total_position(ind)
                current_beta = self.beta[ind]
        
        #Exit last position
        if current != Position.FLAT:
            pnl += current.value * self.calc_pnl_by_trade(ind, current_beta)
            ret *= 1 + pnl/current_position
        return ret - 1
    
    def get_cumm_pnl(self):
        """get_cumm_pnl 
        return the time series of market value (returns)
        """
        
        signals = self.get_signals()
        current = Position.FLAT
        
        mkt_values = np.zeros(len(signals)) #record the mkt value change
        total_position = self.x_test + self.y_test * self.beta
        pnl = 0
        total_pnl = 0
        current_position = 0
        current_beta = 0
        ret = 1
        for ind, p in enumerate(signals):
            
            # Exit current position
            if (current == Position.LONG and p >=0) \
                or (current == Position.SHORT and p <= 0):
                pnl += current.value * self.calc_pnl_by_trade(ind, current_beta)
                total_pnl += current.value * self.calc_pnl_by_trade(ind, current_beta)      
                ret *= 1 + pnl/current_position
                current = Position.FLAT
                
            # Enter new position
            if p == 2 and current != Position.SHORT:
                current = Position.SHORT
                pnl = self.spread[ind] # get cash from short
                total_pnl += pnl
                current_position = total_position[ind]
                current_beta = self.beta[ind]
            elif p == -2 and current != Position.LONG:
                current = Position.LONG
                pnl = -self.spread[ind] # pay cash
                total_pnl += pnl
                current_position = total_position[ind]
                current_beta = self.beta[ind]

            temp_pnl = total_pnl + current.value * self.calc_pnl_by_trade(ind, current_beta)
            mkt_values[ind] = ret * (1 + temp_pnl/current_position)
        
        return mkt_values

    def get_plot(self):
        """get_plot 
        plot the spread in both of the training and testing period
        +- 1 sigma and +- 2 sigma lines on testing period
        """
        def beta(df):
            x, y = df[:,0], df[:,1]
            return y.T @ x / (y.T @ y)
        rolling_beta = self.sample.rolling(self.window, min_periods = 1, method = 'table')\
            .apply(beta, raw = True, engine= 'numba').fillna(0).values[:,0]
        spread = self.sample.iloc[:,0] - rolling_beta * self.sample.iloc[:,1]
        print("working")
        s = spread.rolling(self.window).std()[self.sample.index[-self.test_length:]]
        m = spread.rolling(self.window).mean()[self.sample.index[-self.test_length:]]
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
    """ 
    Calculate the pnl of the bundle of pairs
    data_batch is a dictionary of data_process.TradingData objects, with key as the pair name
    """
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


