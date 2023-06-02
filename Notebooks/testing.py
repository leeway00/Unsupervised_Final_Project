import os
import warnings

import pandas as pd
from model.data_process import *
from model.linear_trading3 import *
from model.reduction import *
from model.diagnosis import *

from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans, OPTICS, SpectralClustering
from sklearn.model_selection import ParameterGrid

import umap
from itertools import product
from tqdm import tqdm

os.chdir("./Unsupervised_Final_Project/")
warnings.filterwarnings('ignore')

datapath = "./data/final_processed/"
results_path = "./results/"

best_settings_df = pd.read_csv(results_path + "best_settings.csv").set_index(['year','Reducing','Clustering'])

daily = pd.read_csv(datapath + 'daily_prices.csv', parse_dates=['date']).sort_values(['date','ticker']).set_index('date')
ratios = pd.read_csv(datapath + 'firm_ratios.csv', parse_dates=['date']).sort_values(['date','ticker']).set_index('date')
sectors = pd.read_csv(datapath + 'sectors.csv', parse_dates=['date']).sort_values(['date','ticker']).set_index('date')
short = pd.read_csv(datapath + 'short_interest_rate.csv', parse_dates=['date']).sort_values(['date','ticker']).set_index('date')
df = daily.merge(ratios, on=['ticker', 'date'], how = 'left')
df = df.merge(short, on =['ticker', 'date'], how = 'left')
df = df.merge(sectors, on=['ticker', 'date'], how = 'left')

tickers = pd.read_csv("./data/sp500_tickers/sp500_historical.csv", parse_dates=['date']).set_index('date')

datagen = DataPrepation(df, tickers)
del tickers, df, daily, ratios, sectors, short

dim_reduction_methods = {
    'PCA': {'name': 'PCA', 'method': PCA, 'params': {'n_components': [2, 3, 4]}},
    'KPCA': {'name': 'KPCA', 'method': KernelPCA, 'params': {'n_components': [2, 3, 4], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}},
    'UMAP': {'name': 'UMAP', 'method': umap.UMAP, 'params': {'n_components': [2, 3, 4], 'n_neighbors': [5, 10, 15]}}
}

clustering_algorithms = {
    'KMeans': {'name': 'KMeans', 'method': KMeans, 'params': {'n_clusters': [5, 10, 15, 30], 'n_init': ['auto']}},
    'OPTICS': {'name': 'OPTICS', 'method': OPTICS, 'params': {'min_samples': [7, 9, 10, 15, 30]}},
    'SpectralClustering': {'name': 'SpectralClustering', 'method': SpectralClustering, 'params': {'n_clusters': [7, 9, 10, 15, 30]}}
}

dim_red_name = ['PCA']
cluster_name = ['KMeans', 'OPTICS', 'SpectralClustering']

def save_test_cluster(year, h = 0):
    
    if h == 0:
        train_start = str(year+1) + '-01-01'
        train_end = str(year+1) + '-12-31'
        test_start = str(year+2) + '-01-01'
        test_end = str(year+2) + '-06-30'
    elif h == 1:
        train_start = str(year+1) + '-07-01'
        train_end = str(year+2) + '-06-30'
        test_start = str(year+2) + '-07-01'
        test_end = str(year+2) + '-12-31'
    elif h == 2:
        train_start = str(year) + '-01-01'
        train_end = str(year) + '-12-31'
        test_start = str(year+2) + '-01-01'
        test_end = str(year+2) + '-12-31'
    elif h == 3:
        train_start = str(year+1) + '-01-01'
        train_end = str(year+1) + '-12-31'
        test_start = str(year+2) + '-01-01'
        test_end = str(year+2) + '-12-31'
    
    X_train, idx, price_norm = datagen.dimension_reduction_preprocess(train_start, train_end)
    price_norm_df = datagen.price_norm(train_start, train_end)
    coint_selector = CointegratedSelection(price_norm_df)
    
    results = pd.DataFrame()
    for dim_red, cluster in tqdm(product(dim_red_name, cluster_name)):
        dim_red_param_num = best_settings_df.loc[(year, dim_red, cluster), 'Reduce_param']
        
        print("Year: {}, Semiannual: {}, Reducing: {}, Clustering: {}".format(year, h, dim_red, cluster))
        dim_red_map = dim_reduction_methods[dim_red]
        reducer = DimensionalityReduction(dim_red_map['method'], dim_red_map['name'], 
                                          X_train, idx, price_norm)
        params  = [i for i in ParameterGrid(dim_red_map['params'])][dim_red_param_num]
        reduced_data = reducer.dimensionality_reduction(params)
        
        print("Reduction Finished")
        cluster_map = clustering_algorithms[cluster]
        cluster_param_num = best_settings_df.loc[(year, dim_red, cluster), 'Cluster_param']
        
        cluster_obj = Clustering(cluster_map['method'], cluster_map['name'], reduced_data)
        params  = [i for i in ParameterGrid(cluster_map['params'])][cluster_param_num]
        clustered = cluster_obj.train_clustering(params, cluster_map['name'])
        
        pairs_list = coint_selector.select_cointegrated_pairs(clustered)
        
        trade_data = datagen.trading_preprocess(pairs_list.pair.values, train_start, train_end, test_start, test_end)
        trade_obj = LinearMulti(trade_data)
        pnls = list(trade_obj.get_pnls(list = True).values())
        
        result = {
            'year': year+2,
            'semiannual': h,
            'Reducing': dim_red,
            'Clusteirng': cluster,
            'pair1': pnls[0],
            'pair2': pnls[1],
            'pair3': pnls[2],
            'pair4': pnls[3],
            'pair5': pnls[4],
            'mean_return': sum(pnls)/5,
            'pair1_name': pairs_list.pair.values[0],
            'pair2_name': pairs_list.pair.values[1],
            'pair3_name': pairs_list.pair.values[2],
            'pair4_name': pairs_list.pair.values[3],
            'pair5_name': pairs_list.pair.values[4],
        }
        
        results = results.append(result, ignore_index=True)
    return results

if __name__ == "__main__":
    first = True
    for y in range(2010, 2019):
        temp1y = save_test_cluster(y, 3)
        temp2y = save_test_cluster(y, 2)
        temp1h = save_test_cluster(y, 0)
        temp2h = save_test_cluster(y, 1)
        if first:
            temp1y.to_csv(results_path + "test_result_SL_1y.csv", index=False)
            temp2y.to_csv(results_path + "test_result_SL_2y.csv", index=False)
            temp1h.to_csv(results_path + "test_result_SL_semi.csv", index=False)
            temp2h.to_csv(results_path + "test_result_SL_semi.csv", index=False, mode = 'a')
            first= False
        else:
            temp1y.to_csv(results_path + "test_result_SL_1y.csv", index=False, header=False, mode = 'a')
            temp2y.to_csv(results_path + "test_result_SL_2y.csv", index=False, header=False, mode = 'a')
            temp1h.to_csv(results_path + "test_result_SL_semi.csv", index=False, header=False, mode = 'a')
        temp2h.to_csv(results_path + "test_result_SL_semi.csv", index=False, header=False, mode = 'a')
            
        
    