
import os
from itertools import combinations

import pickle as pkl
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.stattools import coint

import multiprocess as mp

class DimensionalityReduction:
    def __init__(self, model, name, X_train, idx, price):
        self.model = model
        self.name = name
        self.X_train = X_train
        self.idx = idx
        self.price = price
        self.scaler = StandardScaler()

    def __flatten_data(self, reduced_data):
        merged_df = pd.DataFrame(np.concatenate((self.idx, self.price, reduced_data), axis=1))\
            .set_index([0,1]).stack().unstack(0).dropna(axis=1, how = 'any')
        return merged_df

    def dimensionality_reduction(self, **params: dict):
        reduced_data = self.model(**params).fit_transform(self. X_train)
        merged_df = self.__flatten_data(reduced_data)
        return merged_df
    
    def fit_generator(self, params: dict):
        def gen():
            for p in ParameterGrid(params):
                yield self.dimensionality_reduction(**p)
        return gen()


class Clustering:
    def __init__(self, model, name, df):
        self.model = model
        self.name = name
        self.df = df
        self.tickers = df.columns.tolist()

    def train_clustering(self, param, name):
        clustering_method = self.model(**param)
        cluster_labels = clustering_method.fit_predict(self.df.T)
        labeled_df = pd.DataFrame({'ticker' : self.tickers, 'cluster': cluster_labels})

        # Group the tickers by the assigned cluster labels
        clusters = labeled_df.groupby('cluster')['ticker'].apply(list).rename(name)
        return clusters

    def clustering_param_tuning(self, param_list):
        def gen():
            vals = (str(v) for v in list(param_list.values())[0])
            for p in ParameterGrid(param_list):
                yield p, f'{self.name}_{next(vals)}'
        with mp.Pool(os.cpu_count()) as pool:
            clusterings = pool.starmap(self.train_clustering, iterable = gen())
        return clusterings
    
class CointegratedSelection:
    def __init__(self, normalized_price: pd.DataFrame, significance=0.05):
        self.price = normalized_price
        self.significance = significance
    
    def check_cointegration(self, pair):
        ticker1, ticker2 = pair

        # Perform the cointegration test
        _, p_value, _ = coint(self.price[ticker1], self.price[ticker2])
        return p_value
    
    def __calculate_cointegrations(self, cluster):
        coint_res = {'pair': [], 'p-value': []}
        for i in range(len(cluster)):
            for pair in combinations(cluster.iloc[i], 2):
                coint_res['p-value'].append(self.check_cointegration(pair))
                coint_res['pair'].append(pair)
        pairs = pd.DataFrame(coint_res).sort_values('p-value').reset_index(drop=True)
        return pairs
    
    def select_cointegrated_pairs(self, cluster, stock_num = 5, use_significance = False):
        pairs = self.__calculate_cointegrations(cluster)
        if use_significance:
            return pairs[pairs['p-value'] < self.significance]
        return pairs[:stock_num]