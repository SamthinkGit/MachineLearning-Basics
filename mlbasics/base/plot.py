import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker

from typing import Optional, Any
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

def plot_mt(df: pd.DataFrame):
    """
    Plots a misclassification table for each model.
    :param df: A pandas DataFrame containing performance data of each model
    .. note:: Condider using missclasification_table() from mlbasics.base.results
    """

    df = df.set_index('model').T
    df.plot(kind='bar')
    plt.xlabel('Model')
    plt.ylabel('Label')
    plt.title('Label Missclasification for each Model')
    plt.show()

def plot_learning_evaluation(df: pd.DataFrame):
    """
    Plots a learning table for one model.
    :param df: A pandas DataFrame containing performance data of each model
    :param start: X axis start value
    :param end: X axis end value
    .. note:: For especifying the X axis you must provide <start> and <end> values
    .. note:: Condider using rresults2df() from with LearningModel for building the dataset
    """

    # --- Computing Time Tendency ---
    z = np.polyfit(df.index, df['Time'], 1)
    p = np.poly1d(z)
    
    # --- Building Figure ---
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(df['Accuracy'], 'g-')
    ax1.plot(df['Precision'], 'b-')
    ax2.plot(df['Time'], 'r-')
    ax2.plot(df.index, p(df.index), "r--", label='Trend Line')

    ax1.set_xlabel('X data')
    ax1.set_ylabel('Accuracy and Precision', color='g')
    ax2.set_ylabel('Time', color='r')

    # --- Showing ---
    plt.show()
    
def scatter(X, Y, support_vectors: Optional[Any] = None):
    """
    Plots a scatter plot of the data points. Supports both 2D and 3D data.

    :param X: Data points (features).
    :param Y: Labels or target values.
    :param support_vectors: Support vectors to highlight (optional).
    """

    # --- Building Figure ---
    fig = plt.figure()

    if len(X) == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    # --- Building Points ---
    ax.scatter(*X, c=Y, s=30, cmap=plt.cm.Paired)
    if support_vectors != None:
        ax.scatter(*support_vectors, facecolors='none', s=100, edgecolors='k')
    
    # --- Showing ---
    plt.title('SVC with RBF Kernel: Support Vectors in PCA Space')
    plt.show()

def bar(x, y, title = "Bar Graph"):
    """
    Creates a bar plot.

    :param x: X-axis values.
    :param y: Y-axis values.
    :param title: The title of the plot.
    """

    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.bar(x, y, align='center')
    plt.show()
    

def confusion_mat(y, y_pred):
    """
    Plots a confusion matrix.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    """

    cm = confusion_matrix(y,y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.ylabel('Real Data')
    plt.xlabel('Prediction')
    plt.title('Confusion Matrix')
    plt.show()
