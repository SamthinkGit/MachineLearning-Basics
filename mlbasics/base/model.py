from typing import Optional, Any
from sklearn.model_selection import train_test_split

class BaseModel():

    def __init__( self, X : Any = None, Y : Any = None):

        self.X = X
        self.indim = X.shape

        self.Y = Y
        self.outdim = len(Y)
        self.labels = set(self.Y)
        self.D = len(self.labels)

            

    def split(
            self,
            SEED : Optional[int] = 110,
            SPLIT_SIZE : Optional[float] = 0.333
        ):

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X,
            self.Y,
            test_size=SPLIT_SIZE,
            random_state=SEED
        )

        self.Nx_test = len(self.X_test)
        self.Nx_train = len(self.X_train)

    def fit(self, classifier: Any, use_split: Optional[bool] = False):

        self.classifier = classifier
        if use_split:
            self.classifier.fit(self.X_train, self.Y_train)
        else:
            self.classifier.fit(self.X, self.Y)
    
    def predict(self, X: Optional[Any] = None, use_split: Optional[bool] = False):
        if use_split:
            self.prediction = self.classifier.predict(self.X_test)
        else:
            self.prediction = self.classifier.predict(X)
        return self.prediction

    def print_info(self):

        line = "="*55
        print(line)
        print("Model M: X -> Y")
        print("X: ", self.indim)
        print("Y: ", self.outdim)
        print("Labels: ", self.labels)
        print("Domain:", self.D)
        print(line)

