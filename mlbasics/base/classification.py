from dataclasses import dataclass
from typing import Any
from sklearn import tree

@dataclass
class Classifiers:
    DecisionTree: Any = tree.DecisionTreeClassifier() 