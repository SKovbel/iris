
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

from dataset import Iris

class LinRegModel():
    def __init__(self, random_seed=1):
        self.random_seed = random_seed
        self.linear = LinearRegression()

    def train(self, X_train, y_train):
        self.linear.fit(X_train, y_train)

    def predict(self, X_test):
       return self.linear.predict(X_test)

if __name__ == '__main__':
    # Iris
    iris = Iris()
    X_train, X_test, y_train, y_test = iris.numpy_dataset(test_size=0.2, normilize=True)
    names = iris.speacias_names()

    # Calculate accuracy
    lin_reg  = LinRegModel()
    lin_reg.train(X_train, y_train)
    y_pred = lin_reg.predict(X_test)
    y_pred = iris.array_binary(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Accuracy: {accuracy:.4f}")