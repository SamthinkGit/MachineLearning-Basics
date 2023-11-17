from dataclasses import dataclass
from typing import Any
from sklearn import tree
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB

@dataclass
class SupportedClassifiers:
    DecisionTree: Any = tree.DecisionTreeClassifier() 
    BernoulliNaives: Any = BernoulliNB()
    MultinomialNaives: Any = MultinomialNB()
    GaussianNaives: Any = GaussianNB()