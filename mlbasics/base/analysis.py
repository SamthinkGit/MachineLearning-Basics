from typing import Optional, List, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from mlbasics.config.utils import get_config
from scipy import stats

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlbasics.base.plot as bplot

class DataAnalyzer():
    """
    A class for analyzing and preprocessing datasets.

    Attributes:
        X (np.ndarray): The feature matrix.
        Y (np.ndarray): The target vector.
        df (pd.DataFrame): DataFrame combining X and Y.
        unique_labels (np.ndarray): Unique labels in Y.
        num_labels (int): Number of unique labels.
        num_columns (int): Number of columns in X.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Initializes the DataAnalyzer with data and computes initiali statistics and unified DataFrame

        :param X: The feature matrix.
        :param Y: The target vector.
        """
        # --- Save Atributes ---
        self.X = X
        self.Y = Y

        # --- Build Dataframe ---
        self.df = pd.DataFrame(self.X)
        self.df['Target'] = self.Y

        # --- Compute Statistics ---
        self.unique_labels = self.df['Target'].unique()
        self.num_labels = len(self.unique_labels)
        self.num_columns = self.X.shape[1]

    def preprocess(self, preprocess_list: List[Any]) -> np.ndarray:
        """
        Preprocesses the feature matrix X using a list of preprocessing steps.

        :param preprocess_list: A list of preprocessing steps.
        :return: The preprocessed feature matrix.

        .. note::
            This function will change the dataset inside DataAnalyzer with the preprocessed one
        """

        # --- Build preprocessing Pipeline ---
        pipeline_steps = [(f'preprocessor_{i}', p) for i, p in enumerate(preprocess_list)]
        preprocess = Pipeline(steps=pipeline_steps)

        # --- Run and Save Preprocessed data ---
        X = preprocess.fit_transform(self.X)
        self.__init__(X,self.Y)

        return X
        
    def plt_distribution(self) -> None:
        """
        Plots the distribution of X.
        """
        config = get_config()
        plt.hist(self.X, bins=config['plot']['resolution'], edgecolor='black')
        plt.title("Distribution of Dataset")
        plt.show()
    
    def plt_target_distribution(self) -> None:
        """
        Plots the distribution of X across different labels in Y.
        """

        # --- Build Figure ---
        fig, axs = plt.subplots(self.num_labels, 1, figsize=(10, 30))

        # --- Compute Distribution for each label ---
        for i in range(self.num_labels):
            subset = self.df[self.df['Target'] == i]
            means = subset.mean()

            axs[i].plot(means[:-1])
            axs[i].set_title(i)

        # --- Show ---
        plt.tight_layout()
        plt.show()
        
    def sanitize(self) -> None:
        """
        Removes columns in X that have the same value across all rows.
        """
        same_value_columns_idx = (self.X == self.X[0]).all(axis=0)
        X = self.X[:, ~same_value_columns_idx]
        self.__init__(X, self.Y)
        

    def filter_outliers(self, percentile: int = 95 ) -> Any:
        """
        Filters outliers from the dataset based on KNN distance.
        :param percentile: The percentile to use as a threshold for outlier detection.
        .. note::
            This function will change the dataset inside DataAnalyzer with the preprocessed one
        """
        
        config = get_config()

        # --- Computing Outliers with KNN ---
        nbrs = NearestNeighbors(n_neighbors=config['model']['n_neighbors']).fit(self.X)
        distances, indices = nbrs.kneighbors(self.X)
        knn_score = distances[:, 1:].mean(axis=1)
        threshold = np.percentile(knn_score, percentile)

        # --- Removing Outliers ---
        outliers = knn_score > threshold
        X_filtered = self.X[~outliers]
        Y_filtered = self.Y[~outliers]

        # --- Saving Atributes ---
        self.__init__(X_filtered, Y_filtered)


    def filter_peaks(self, contamination=0.05) -> None:
        """
        Filters peaks in the dataset using Isolation Forest.
        :param contamination: The contamination factor for Isolation Forest.
        """

        # --- Finding Peaks ---
        iso_forest = IsolationForest(contamination=contamination)
        iso_forest.fit(self.X)
        preds = iso_forest.predict(self.X)
        
        # --- Removing ---
        X_filtered = self.X[preds == 1, :]
        Y_filtered = self.Y[preds == 1]

        # --- Saving ---
        self.__init__(X_filtered, Y_filtered)
    
    def compute_feature_importances(self) -> np.ndarray:
        """
        Computes feature importances using a DecisionTreeClassifier.
        :return: Array of feature importances.
        """

        model = DecisionTreeClassifier()
        model.fit(self.X, self.Y)
        return model.feature_importances_



        
class ModelAnalyzer():

    """
    A class for analyzing machine learning models.

    Attributes:
        model: The machine learning model.
        x (np.ndarray): The feature matrix.
        y_real (np.ndarray): The actual target values.
        y_pred (np.ndarray): The predicted target values.
    """
    def __init__(self, model, x: np.ndarray, y_real: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Initializes the ModelAnalyzer with a model and data.
        """
        self.model = model 
        self.x = x 
        self.y_real = y_real,
        self.y_pred = y_pred,

    def plot_results(self, dim):
        """
        Plots the results of the model predictions using PCA for dimensionality reduction.
        :param dim: The number of dimensions for PCA.
        """

        pca = PCA(n_components=dim)

        X_pca = pca.fit_transform(self.x)
        x_plt = [X_pca[:,i] for i in range(dim)]

        bplot.scatter(X=x_plt, Y=self.y_pred)