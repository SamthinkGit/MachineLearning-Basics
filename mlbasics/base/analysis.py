from typing import Optional, List, Any

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer 
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.neighbors import NearestNeighbors
from mlbasics.config.utils import get_config
from scipy import stats

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataAnalyzer():

    def __init__(self,X,Y):
        self.X = X
        self.Y = Y

        self.df = pd.DataFrame(self.X)
        self.df['Target'] = self.Y

        self.unique_labels = self.df['Target'].unique()
        self.num_labels = len(self.unique_labels)
        self.num_columns = self.X.shape[1]

    def preprocess(self, preprocess_list: List[Any]):

        pipeline_steps = []
        for i, p in enumerate(preprocess_list):
            pipeline_steps.append((f'preprocessor_{i}', p))

        preprocess = Pipeline(steps=pipeline_steps)
        X = preprocess.fit_transform(self.X)
        self.__init__(X,self.Y)

        return X
        
    def plt_distribution(self) -> Any:

        config = get_config()
        plt.hist(self.X, bins=config['plot']['resolution'], edgecolor='black')
        plt.title("Distribution of Dataset")
        plt.show()
    
    def plt_target_distribution(self):

        fig, axs = plt.subplots(self.num_labels, 1, figsize=(10, 30))

        for i in range(self.num_labels):
            subset = self.df[self.df['Target'] == i]
            means = subset.mean()

            axs[i].plot(means[:-1])
            axs[i].set_title(i)

        plt.tight_layout()
        plt.show()
        
    def sanitize(self):
        same_value_columns_idx = (self.X == self.X[0]).all(axis=0)
        X = self.X[:, ~same_value_columns_idx]
        self.__init__(X, self.Y)
        

    def filter_outliers(self, percentile: int = 95 ) -> Any:

        config = get_config()

        nbrs = NearestNeighbors(n_neighbors=config['model']['n_neighbors']).fit(self.X)
        distances, indices = nbrs.kneighbors(self.X)
        knn_score = distances[:, 1:].mean(axis=1)
        threshold = np.percentile(knn_score, percentile)

        outliers = knn_score > threshold
        X_filtered = self.X[~outliers]
        Y_filtered = self.Y[~outliers]
        self.__init__(X_filtered, Y_filtered)


    def filter_peaks(self, contamination=0.05):

        iso_forest = IsolationForest(contamination=contamination)
        iso_forest.fit(self.X)
        preds = iso_forest.predict(self.X)
        
        X_filtered = self.X[preds == 1, :]
        Y_filtered = self.Y[preds == 1]

        self.__init__(X_filtered, Y_filtered)
        