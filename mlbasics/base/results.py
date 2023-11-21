from typing import Any, Optional, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc
    )

from mlbasics.config.utils import get_config
import matplotlib.pyplot as plt
import mlbasics.base.plot as bplot
import numpy as np
import pandas as pd
import time

class ClassificationResults():
    
    def __init__(
        self,
        name: Optional[str] = None,
        x: Any = None,
        y_real: Optional[Any] = None,
        y_predicted: Any = None,
        labels: Any = None,
        training_time: time.time = None,
        kfold_score: Optional[Any] = None,
        y_scores: Optional[np.array] = None,
        ) -> None:
        self.name = name
        self.x = x
        self.y_real = y_real
        self.y_predicted = y_predicted
        self.training_time = training_time
        self.kfold_score = kfold_score
        self.y_scores = y_scores
        self.labels = labels
        self.y_scores = y_scores

        if y_scores is not None:
            self.auc = roc_auc_score(self.y_real, y_scores, multi_class='ovr')
            self.fpr = dict()
            self.tpr = dict()
            self.roc = dict()
            self.compute_roc()

        if y_real is not None:
            self.accuracy = accuracy_score(y_real, y_predicted)
            self.precision = precision_score(y_real, y_predicted, average='weighted', zero_division=0)
            self.recall = recall_score(y_real, y_predicted, average='weighted')
            self.f1 = f1_score(y_real, y_predicted, average='weighted')

        
    def plot_confusion_mat(self):
        bplot.confusion_mat(self.y_real, self.y_predicted)

    def plot_roc_curves(self):
        plt.figure()
        for i in range(len(self.labels)):
            plt.plot(self.fpr[i], self.tpr[i], label=f'ROC curve (area = {self.roc[i]:0.2f}) for class {i}')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC')
        plt.legend(loc="lower right")
        plt.show()

    def print_training_time(self):
        print("Training Time: ", self.training_time)
    
    def print_classification_report(self):
        print(classification_report(self.y_real, self.y_predicted))
    
    def compute_roc(self):
        config = get_config()
        for i in range(len(self.labels)):
            self.fpr[i], self.tpr[i], _ = roc_curve(self.y_real == i, self.y_scores[:, i])
            self.roc[i] = auc(self.fpr[i], self.tpr[i])

        self.round_roc = dict()
        for key, value in self.roc.items():
            self.round_roc[key] = round(value, config['plot']['round'])

def results2df(results: List[ClassificationResults]) -> pd.DataFrame:
    df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1", "Time", "K-Fold Score",
                               "AUC", "ROC"])
    rnd =  get_config()['plot']['round']
    for model in results:
        df.loc[len(df.index)] = [
            model.name,
            round(model.accuracy, rnd),
            round(model.precision, rnd),
            round(model.recall, rnd),
            round(model.f1, rnd),
            round(model.training_time, rnd),
            round(model.kfold_score.mean(), rnd) if model.kfold_score is not None else "-",
            round(model.auc, rnd) if model.y_scores is not None else "-",
            # TODO: Show this as separated DF
            model.round_roc if model.y_scores is not None else "-"
            
            ]
    
    return df


def missclasification_table(results: List[ClassificationResults], labels: List[Any], show_proportion: Optional[bool] = True) -> pd.DataFrame:

    df = pd.DataFrame(columns = ["model"] + labels)

    for model in results:
        cm = confusion_matrix(model.y_real,model.y_predicted)
        N = cm.shape[0]
        error = np.zeros(N)

        for i in range(N):
            misses = np.sum(cm[:, i]) - cm[i, i] 
            total = np.sum(cm[:,i])

            if show_proportion:
                error[i] = round(misses / total, 3)
            else:
                error[i] = misses
            

        df.loc[len(df.index)] = [model.name] + list(error)

    return df 