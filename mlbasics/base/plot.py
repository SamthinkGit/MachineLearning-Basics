import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

def pca_scatter(X,Y):
 
    # Reducir la dimensión de los datos a 2 componentes
    pca = PCA(n_components=2)
    X_r = pca.fit_transform(X)

    # Plotear los dos componentes principales
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