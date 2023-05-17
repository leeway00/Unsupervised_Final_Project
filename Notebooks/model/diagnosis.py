import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





class Diagnosis:
    
    @staticmethod
    def plot_clustered(price_norm, cluster_obj):
        n = len(cluster_obj)
        fig, ax = plt.subplots(figsize=(15, 10), nrows=n//3+1, ncols=3)
        for i in range(len(cluster_obj)):
            print(f"{i}th cluster: {len(cluster_obj.iloc[i])}")
            price_norm[cluster_obj.iloc[i]].plot(legend = False, ax = ax[i//3, i%3])
        # return fig
    
    @staticmethod
    def plot_pairs(price_norm, pairs):
        n = len(pairs)
        fig, ax = plt.subplots(figsize=(15, 10), nrows=n//2+1, ncols=2)
        for i in range(pairs):
            print(f"{i}th pair: {pairs.pair.iloc[i]}")
            price_norm[list(pairs.pair.iloc[i])].plot(legend = False, ax = ax[i//2, i%2])
        # return fig
    
    @staticmethod
    def plot_test_price(price_norm, pairs):
        fig, ax = plt.subplots(figsize=(15, 10), nrows=3, ncols=2)
        for i in range(5):
            print(f"{i}th pair: {pairs.pair.iloc[i]}")
            price_norm[list(pairs.pair.iloc[i])].plot(legend = False, ax = ax[i//2, i%2])