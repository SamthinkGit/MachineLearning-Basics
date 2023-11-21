import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from typing import Optional
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

def plot_mt(df: pd.DataFrame):

    df = df.set_index('model').T
    df.plot(kind='bar')
    plt.xlabel('Model')
    plt.ylabel('Label')
    plt.title('Label Missclasification for each Model')
    plt.show()

def pca_scatter(X,Y):
    pca = PCA(n_components=2)
    X_r = pca.fit_transform(X)

    plt.scatter(
        X_r[:, 0],
        X_r[:, 1],
        c=Y,
        cmap='viridis',
        s=20, 
        alpha=0.7,
        marker='x'
        )
    plt.title('Scatter plot PCA')
    plt.colorbar()
    plt.show()

def confusion_mat(y, y_pred):
    cm = confusion_matrix(y,y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.ylabel('Real Data')
    plt.xlabel('Prediction')
    plt.title('Confusion Matrix')
    plt.show()
