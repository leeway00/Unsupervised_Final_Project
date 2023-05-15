
import pandas as pd
import numpy as np
import pickle as pkl
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import ParameterGrid

import multiprocess as mp

class DimensionalityReduction:
    def __init__(self, df, date_range, formation_period):
        self.df = df
        self.date_range = date_range
        self.formation_period = formation_period
        self.scaler = StandardScaler()

    def __subset_data(self, period_start, period_end):
        # period_end = period_start + pd.DateOffset(months=int(formation_period[:-1]))
        df_period = self.df.loc[period_start.strftime('%Y-%m'):period_end.strftime('%Y-%m')]
        to_drop = df_period.loc[df_period.isna().any(axis=1)]['ticker'].unique()
        df_period = df_period.loc[~df_period['ticker'].isin(to_drop)]
        return df_period

    def preprocess(self, period_start, period_end):
        # Preprocessing steps
        df_period = self.__subset_data(period_start, period_end)
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

    def __flatten_data(self, reduced_data, idx):
        merged_df = pd.DataFrame(np.concatenate((idx, reduced_data), axis=1))\
            .set_index([0,1]).stack().unstack(0).dropna(axis=1, how = 'any')
        return merged_df

    def dimensionality_reduction(self, X_train, idx, model, n_components):
        reduced_data = model(n_components = n_components).fit_transform(X_train)
        merged_df = self.__flatten_data(reduced_data, idx)
        return merged_df


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