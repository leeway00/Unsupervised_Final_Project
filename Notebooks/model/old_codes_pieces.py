import numpy as np
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt



class LinearSignal_No_Rolling:
    def __init__(self, pair, data_obj, window = 200):
        self.pair = pair
        self.get_data(data_obj)
        self.test_length = len(self.x_test)
        self.beta = self.get_ratio()
        self.sigma, self.mean = self.pre_calculation()


    def get_data(self, data_obj):
        self.sample = data_obj.sample
        self.train_period = data_obj.train_period
        self.test_period = data_obj.test_period
        
        train = self.sample.loc[self.train_period[0]: self.train_period[1]].values
        self.x_train = train[:,0]
        self.y_train = train[:,1]
        test = self.sample.loc[self.test_period[0]: self.test_period[1]].values
        self.x_test = test[:,0]
        self.y_test = test[:,1]
        pass
    
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