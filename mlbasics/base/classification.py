from dataclasses import dataclass
from typing import Any, List, Optional
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron
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
        return LogisticRegression(multi_class='multinomial', max_iter=config['model']['max_iterations'])

    @staticmethod
    def KNeighbors() -> KNeighborsClassifier:
        config = get_config()
        return KNeighborsClassifier(n_neighbors=config['model']['n_neighbors'])

    @staticmethod
    def LinearSVC() -> SVC:
        return SVC(kernel='linear', probability=True)

    @staticmethod
    def KernelizedSVC() -> SVC:
        return SVC(kernel='rbf', probability=True)

    @staticmethod
    def PolySVC() -> SVC:
        return SVC(kernel='poly', probability=True)

    @staticmethod
    def PerceptronClassifier() -> Perceptron:
        Perceptron.predict_proba = lambda self, y: None 
        return Perceptron()

    @staticmethod
    def FisherLineal() -> LinearDiscriminantAnalysis:
        return LinearDiscriminantAnalysis()

def get_all_supported_classifiers(skip_list: Optional[List[str]] = []):
    classifiers = {}
    for clf in inspect.getmembers(SupportedClassifiers, inspect.isfunction):
        if not clf[0].startswith('__') and clf[0] not in skip_list:
            classifiers[clf[0]] = clf[1]()
    return classifiers