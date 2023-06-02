
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
    """ Reduce the dimension of the data
    
    Initialize:
        model: sklearn object: the dimensionality reduction model
        name: string: the name of the model
        X_train: np.array: the data to be reduced
        idx: np.array: the index of the data
        price: np.array: the nomalized price of the data
    """
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

    def dimensionality_reduction(self, params: dict):
        # Fit the model and transform the data
        # If you have a targeting parameter, you can use this function directly
        reduced_data = self.model(**params).fit_transform(self. X_train)
        merged_df = self.__flatten_data(reduced_data)
        return merged_df
    
    def fit_generator(self, params: dict):
        """fit_generator
        Generate the reduced data with different parameters
        It returns a generator of reduced data
        for example:
            for reduced_data in fit_generator(params):
                # do something with reduced_data
        """
        def gen():
            for p in ParameterGrid(params):
                yield self.dimensionality_reduction(p)
        return gen()


class Clustering:
    """
    Clustering the data given the model and parameters
    
    Initialize:
        model: sklearn object: the clustering model
        name: string: the name of the model
        df: pd.DataFrame: dimension reduced data
            - each columns corresponds to a ticker
            - values are vectorized data for each ticker
            - each row corresponds to date x reduced dimension
    """
    def __init__(self, model, name, df):
        self.model = model
        self.name = name
        self.df = df
        self.tickers = df.columns.tolist()

    def train_clustering(self, param, name = 'cluster'):
        """train_clustering
        You can use this function directly if you have a targeting parameter
        
        param: dict: the parameters for the clustering model
            - should exactly match with the model requirement
            - for example: {'n_clusters': 5}
        name: string: the name of the cluster model
            - to distinguish the model result from different parameters
            - Doesn't need to be specified if you are using the targeting parameter
        """
        
        clustering_method = self.model(**param)
        cluster_labels = clustering_method.fit_predict(self.df.T)
        labeled_df = pd.DataFrame({'ticker' : self.tickers, 'cluster': cluster_labels})

        # Group the tickers by the assigned cluster labels
        clusters = labeled_df.groupby('cluster')['ticker'].apply(list).rename(name)
        return clusters

    def clustering_param_tuning(self, param_list):
        """
        Fit several clustering models at once
        This uses multiprocessing to fit the different hyperparameters in parallel
        """
        def gen():
            vals = (str(v) for v in list(param_list.values())[0])
            for p in ParameterGrid(param_list):
                yield p, f'{self.name}_{next(vals)}'
        with mp.Pool(os.cpu_count()) as pool:
            clusterings = pool.starmap(self.train_clustering, iterable = gen())
        return clusterings
    
class CointegratedSelection:
    """ 
    Select the cointegrated pairs from the clusters
    
    self.select_cointrated_pairs: use this method to get the cointegrated pairs
    """
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