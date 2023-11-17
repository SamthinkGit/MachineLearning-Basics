from typing import Any, Optional, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
    )

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
        training_time: time.time = None
        ) -> None:
        self.name = name
        self.x = x
        self.y_real = y_real
        self.y_predicted = y_predicted
        self.training_time = training_time

        if y_real is not None:
            self.accuracy = accuracy_score(y_real, y_predicted)
            self.precision = precision_score(y_real, y_predicted, average='weighted')
            self.recall = recall_score(y_real, y_predicted, average='weighted')
            self.f1 = f1_score(y_real, y_predicted, average='weighted')

    def plot_confusion_mat(self):
        bplot.confusion_mat(self.y_real, self.y_predicted)

    def print_training_time(self):
        print("Training Time: ", self.training_time)
    
    def print_classification_report(self):
        print(classification_report(self.y_real, self.y_predicted))
        

def results2df(results: List[ClassificationResults]) -> pd.DataFrame:
    df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1", "Time"])

    for model in results:
        df.loc[len(df.index)] = [
            model.name,
            round(model.accuracy, 3),
            round(model.precision, 3),
            round(model.recall, 3),
            round(model.f1, 3),
            round(model.training_time, 3)
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