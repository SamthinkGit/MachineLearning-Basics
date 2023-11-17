from typing import Optional, Any, Tuple, Dict, List
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import time

from mlbasics.data.load_data import load_data
from mlbasics.base.results import ClassificationResults

class BaseModel:
    """
    A base model class that provides basic functionalities to handle machine learning models.
    """

    def __init__(self, X: Optional[Any] = None, Y: Optional[Any] = None):
        """
        Initializes the BaseModel with data and computes dimensions.
        :param X: Input features.
        :param Y: Target values.
        """
        self.X = X
        self.indim = X.shape[1]  # Assumes X is a 2D array-like structure

        self.Y = Y
        self.outdim = Y.shape[0] if len(Y.shape) > 0 else 1
        self.labels = set(Y)
        self.D = len(self.labels)
    

    def split(self, seed: int = 110, split_size: float = 0.333):
        """
        Splits the dataset into training and testing sets.
        :param seed: Random seed for reproducibility.
        :param split_size: Proportion of the dataset to include in the test split.
        """
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X,
            self.Y,
            test_size=split_size,
            random_state=seed
        )

    def fit(self, classifier: Any, use_split: bool = False):
        """
        Fits the classifier to the training data.
        :param classifier: The machine learning classifier to train.
        :param use_split: Whether to use the split dataset for training.
        """
        self.classifier = classifier
        if use_split: 
            self.classifier.fit(self.X_train, self.Y_train)
        else:
            self.classifier.fit(self.X, self.Y)

    def predict(self, X: Optional[Any] = None, use_split: bool = False) -> Any:
        """
        Makes predictions using the classifier.
        :param X: Input features to make predictions on. If None and use_split is True, use self.X_test.
        :param use_split: Whether to use the testing set for predictions.
        :return: Predicted values.
        """
        if use_split:
            return self.classifier.predict(self.X_test)
        else:
            return self.classifier.predict(X)

    def print_info(self):
        """
        Prints information about the model and dataset.
        """
        line = "=" * 55
        print(line)
        print("Model M: X -> Y")
        print(f"X: {self.indim}")
        print(f"Y: {self.outdim}")
        print(f"Labels: {self.labels}")
        print(f"Domain: {self.D}")
        print(line)


class CompactModel(BaseModel):
    """
    A compact model class derived from BaseModel, providing a streamlined process to load data,
    split, train, predict, and plot results.
    """

    def __init__(
        self,
        dataset: Optional[str] = None,
        X: Optional[Any] = None,
        Y: Optional[Any] = None,
        use_split: bool = True,
        split_size: float = 0.3,
        classifiers: Dict[str, Any] = {}
    ):
        """
        Initializes the CompactModel with a dataset or data, and sets up the environment for training and evaluation.
        You can either pass the input as a string in "dataset" or directly to "X,Y"
        :param dataset: Name of the dataset to load.
        :param X: Input features.
        :param Y: Target values.
        :param use_split: Whether to split the dataset.
        :param split_size: Size of the split for the test set.
        :param classifiers: A dictionary with the names and algorithms to use for training
        :param show_results: Whether to print classification report.
        :param plot_results: Whether to plot confusion matrix.
        """
        if dataset:
            X, Y = load_data(dataset)

        super().__init__(X, Y)

        if use_split:
            self.split(split_size=split_size)

        self.use_split = use_split
        self.classifiers = classifiers

    def run(self) -> List[ClassificationResults]:

        """
        Run all the models with the parameters given at initialization
        """

        results = []

        for name, model in self.classifiers.items():
        
            timer = time.time()
            self.fit(classifier=model,use_split=self.use_split)
            training_time = time.time() - timer
            prediction = self.predict(use_split=self.use_split)

            output = ClassificationResults(
                name = name,
                x = self.X_test if self.split else self.X,
                y_real= self.Y_test if self.split else self.Y,
                y_predicted = prediction,
                training_time = training_time
            )
            results.append(output)

        return results