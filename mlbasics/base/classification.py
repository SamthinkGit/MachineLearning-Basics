from dataclasses import dataclass
from typing import Any, List, Optional
from sklearn import tree
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mlbasics.config.utils import get_config

import inspect

@dataclass
class SupportedClassifiers:

    @staticmethod
    def DecisionTree() -> tree.DecisionTreeClassifier:
        return tree.DecisionTreeClassifier()

    @staticmethod
    def BernoulliNaives() -> BernoulliNB:
        return BernoulliNB()

    @staticmethod
    def MultinomialNaives() -> MultinomialNB:
        return MultinomialNB()

    @staticmethod
    def GaussianNaives() -> GaussianNB:
        return GaussianNB()

    @staticmethod
    def LogisticMultinomialRegression() -> LogisticRegression:
        config = get_config()
        log = LogisticRegression(multi_class='multinomial', max_iter=config['model']['max_iterations'])
        return make_pipeline(StandardScaler(), log)

def get_all_supported_classifiers(skip_list: Optional[List[str]] = []):
    classifiers = {}
    for clf in inspect.getmembers(SupportedClassifiers, inspect.isfunction):
        if not clf[0].startswith('__') and clf[0] not in skip_list:
            classifiers[clf[0]] = clf[1]()
    return classifiers