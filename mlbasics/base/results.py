from typing import Any, Optional, List
from prettytable import PrettyTable
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
    )

import mlbasics.base.plot as bplot
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
        

def results2table(results: List[ClassificationResults] = []):
    table = PrettyTable()
    table.field_names = ["Model", "accuracy", "precision", "recall", "f1", "time"]
    for obj in results:
        table.add_row([
            obj.name, 
            round(obj.accuracy,3), 
            round(obj.precision,3),
            round(obj.recall,3), 
            round(obj.f1,3),
            round(obj.training_time,3)
            ])
    
    return table
    