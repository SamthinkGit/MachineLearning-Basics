import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from typing import Optional, Any
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

def plot_mt(df: pd.DataFrame):

    df = df.set_index('model').T
    df.plot(kind='bar')
    plt.xlabel('Model')
    plt.ylabel('Label')
    plt.title('Label Missclasification for each Model')
    plt.show()

def scatter(X, Y, support_vectors: Optional[Any] = None):

        fig = plt.figure()

        if len(X) == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        ax.scatter(*X, c=Y, s=30, cmap=plt.cm.Paired)

        if support_vectors != None:
            ax.scatter(*support_vectors, facecolors='none', s=100, edgecolors='k')
        
        plt.title('SVC with RBF Kernel: Support Vectors in PCA Space')
        plt.show()

def bar(x, y, title = "Bar Graph"):

    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.bar(x, y, align='center')
    plt.show()
    

def confusion_mat(y, y_pred):
    cm = confusion_matrix(y,y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.ylabel('Real Data')
    plt.xlabel('Prediction')
    plt.title('Confusion Matrix')
    plt.show()
