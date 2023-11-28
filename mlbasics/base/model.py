from typing import Optional, Any, Tuple, Dict, List
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
import time

from tqdm import tqdm
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
        self.X_test = None
        self.Y_test = None
        self.indim = X.shape[1]  # Assumes X is a 2D array-like structure

        self.Y = Y
        self.outdim = Y.shape[0] if len(Y.shape) > 0 else 1
        self.labels = list(set(Y))
        self.D = len(self.labels)
        self.kfold_scores = None
    

    def split(self, seed: int = 110, split_size: float = 0.333, X: Optional[Any] = None, Y: Optional[Any] = None):
        """
        Splits the dataset into training and testing sets.
        :param seed: Random seed for reproducibility.
        :param split_size: Proportion of the dataset to include in the test split.
        :param X: (Optional) Used to train X with different values
        :param X: (Optional) Used to train Y with different values
        """

        if X is not None and Y is not None:
            x_clf = X
            y_clf = Y
        else:
            x_clf = self.X
            y_clf = self.Y

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            x_clf,
            y_clf,
            test_size=split_size,
            random_state=seed
        )

    def fit(self, classifier: Any, use_split: bool = False, kfold: KFold = None):
        """
        Fits the classifier to the training data.
        :param classifier: The machine learning classifier to train.
        :param use_split: Whether to use the split dataset for training.
        """
        self.classifier = classifier
        if use_split: 
            x_clf = self.X_train
            y_clf = self.Y_train
        else:
            x_clf = self.X
            y_clf = self.Y

        if kfold is not None:
            self.kfold_scores = cross_val_score(self.classifier, x_clf, y_clf, cv=kfold)

        self.classifier.fit(x_clf, y_clf)

    def predict(self, X: Optional[Any] = None, use_split: bool = False) -> Any:
        """
        Makes predictions using the classifier.
        :param X: Input features to make predictions on. If None and use_split is True, use self.X_test.
        :param use_split: Whether to use the testing set for predictions.
        :return: Predicted values.
        """
        if use_split:
            return self.classifier.predict(self.X_test)
        elif X is not None:
            return self.classifier.predict(X)
        else:
            return self.classifier.predict(self.X)
            

    def predict_proba(self, X: Optional[Any] = None, use_split: bool = False) -> Any:
        """
        Makes predictions using the classifier.
        :return: Predicted probabilities of all values.
        """
        if use_split:
            return self.classifier.predict_proba(self.X_test)
        elif X is not None:
            return self.classifier.predict_proba(X)
        else:
            return self.classifier.predict_proba(self.X)
    
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
        preprocessing: Optional[List[Any]] = None,
        use_split: bool = True,
        split_size: float = 0.3,
        classifiers: Dict[str, Any] = {},
        kfold: KFold = None,
        compute_probs: Optional[bool] = False,
        seed: int = 110
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
        :param kfold: A Kfold object to test the classifiers
        :param compute_probs: Force the algorithm to return the probabilities of X in results
        :param show_results: Whether to print classification report.
        :param plot_results: Whether to plot confusion matrix.
        """
        # --- Load Dataset ---
        if dataset:
            X, Y = load_data(dataset)

        # --- Add Preprocessing (If settled) ---
        if preprocessing is not None:
            for p in preprocessing:
                X = p.fit_transform(X) 

        super().__init__(X, Y)

        # --- Split (If settled) ---
        if use_split:
            self.split(split_size=split_size, seed=seed)

        self.use_split = use_split

        # --- Save variables ---
        self.classifiers = classifiers
        self.kfold = kfold
        self.preprocessing = preprocessing
        self.compute_probs = compute_probs




    def run(self) -> Dict[str,ClassificationResults]:

        """
        Run all the models with the parameters given at initialization
        """

        results = {}
        
        # --- For each clf ---
        with tqdm(self.classifiers.items(), desc="Training Models") as t:
            for name, model in t:
        
                # --- Initialice Run ---
                t.set_description(name)
                timer = time.time()

                # --- Train clf ---
                self.fit(classifier=model,use_split=self.use_split, kfold=self.kfold)
                training_time = time.time() - timer

                # --- Predict new X ---
                prediction = self.predict(use_split=self.use_split)

                # --- Save Results ---
                output = ClassificationResults(
                    name = name,
                    x = self.X_test if self.use_split else self.X,
                    y_real = self.Y_test if self.use_split else self.Y,
                    model=model,
                    y_predicted = prediction,
                    labels=self.labels,
                    training_time = training_time,
                    kfold_score=self.kfold_scores,
                    y_scores=self.predict_proba(use_split=self.use_split) if self.compute_probs else None,
                )

                # --- Prepare next Run ---
                results[name] = output
                t.write(f"✅ Completed: {name}")

        return results
    
class LearningModel(BaseModel):
    """
    A machine learning model that supports iterative training (for hyperparameter searching).

    Attributes:
        clfs: A list of classifiers to be used in the training process.
        use_split: Boolean indicating whether to use data splitting.
        split_size: The proportion of the dataset to include in the test split.
        seed: The seed for random number generation.
        steps: Number of steps or iterations for training.
        preprocessing: Preprocessing steps to apply to the data.
        classifiers: A list of classifiers configurations.
        kfold: KFold cross-validator.
        compute_probs: Whether to compute prediction probabilities.
        results: Dictionary to store the results after each iteration.
        iteration: Counter for the current iteration.
    """

    def __init__(
        self,
        dataset: str | None = None,
        X: Any | None = None,
        Y: Any | None = None,
        preprocessing: Any = None,
        split_size: float = 0.3,
        classifier: Dict[str,Any] = None,
        kfold: KFold = None,
        compute_probs: bool | None = False,
        seed: int = 110,
        steps: int = 30
    ):

        # --- Building Multiple Classifiers as a Dict ---
        clfs = []
        for i in range(steps):
            clfs.append({"name": f'{classifier["name"]}{i}', "model": classifier["model"]})
        self.clfs = clfs

        # --- Load Dataset ---
        if dataset:
            X, Y = load_data(dataset)

        super().__init__(X,Y)

        # --- Save variables ---
        self.use_split = True
        self.split_size = split_size
        self.seed = seed
        self.steps = steps
        self.preprocessing = preprocessing 
        self.classifiers = clfs
        self.kfold = kfold
        self.compute_probs = compute_probs
        self.results = {}
        self.iteration = 0

    def step(self):
        """
        Executes a single training step, including preprocessing, training, and result storing.

        .. note:: You may want to change the attributes of the model between each step, if not,
            consider using CompactModel
        """

        timer = time.time()

        # --- Preprocessing ---
        if self.preprocessing is not None:
            X_clf = self.X
            Y_clf = self.Y
            for p in self.preprocessing:
                X_clf = p.fit_transform(X_clf) 
            self.split(split_size=self.split_size, seed=self.seed, X=X_clf, Y=Y_clf)

        # --- Train clf ---
        self.fit(
            classifier=self.clfs[self.iteration]["model"],
            use_split=self.use_split,
            kfold=self.kfold,
            )
        
        training_time = time.time() - timer

        # --- Predict new X ---
        prediction = self.predict(use_split=self.use_split)

        # --- Save Results ---
        output = ClassificationResults(
            name=self.clfs[self.iteration]["name"],
            x=self.X_test,
            y_real=self.Y_test,
            model=self.clfs[self.iteration]["model"],
            y_predicted=prediction,
            labels=self.labels,
            training_time=training_time,
            kfold_score=self.kfold_scores,
            y_scores=self.predict_proba(use_split=self.use_split) if self.compute_probs else None,
        )

        # --- Prepare next Run ---
        self.results[self.clfs[self.iteration]["name"]] = output
        self.iteration += 1