from typing import Optional, Any, Tuple
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from mlbasics.data.load_data import load_data
import mlbasics.base.plot as bplot


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
        classifier: Any = None,
        show_results: bool = True,
        plot_results: bool = True
    ):
        """
        Initializes the CompactModel with a dataset or data, and sets up the environment for training and evaluation.
        :param dataset: Name of the dataset to load.
        :param X: Input features.
        :param Y: Target values.
        :param use_split: Whether to split the dataset.
        :param split_size: Size of the split for the test set.
        :param classifier: The machine learning classifier to train.
        :param show_results: Whether to print classification report.
        :param plot_results: Whether to plot confusion matrix.
        """
        if dataset:
            X, Y = load_data(dataset)

        super().__init__(X, Y)

        if use_split:
            self.split(split_size=split_size)

        self.use_split = use_split
        self.classifier = classifier
        self.show_results = show_results
        self.plot_results = plot_results

    def run(self) -> Any:
        """
        Run all the model with the parameters given at initialization
        """
        
        self.fit(classifier=self.classifier,use_split=self.use_split)
        prediction = self.predict(use_split=self.use_split)
    
        if self.show_results:
            print(classification_report(self.Y_test, prediction))

        if self.plot_results:
            bplot.confusion_mat(self.Y_test, prediction)
        
        return self.prediction