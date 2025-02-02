
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from dataset import Iris

# Logistic Regression: Primarily used for binary classification problems, where the goal is to predict one of two classes.
class LogRegModel():
    def __init__(self, random_seed=1):
        self.random_seed = random_seed
        self.logistic = LogisticRegression()

    def train(self, X_train, y_train, epochs=100):
        self.logistic = LogisticRegression(max_iter=epochs)
        self.logistic.fit(X_train, y_train)

    def predict(self, X_test):
        return self.logistic.predict(X_test)

if __name__ == '__main__':
    epochs = 1000

    # Iris
    iris = Iris()
    X_train, X_test, y_train, y_test = iris.numpy_dataset(test_size=0.2, one_hot_y=False, normilize=True)
    names = iris.speacias_names()

    # Calculate accuracy
    log_reg = LogRegModel()
    log_reg.train(X_train, y_train, epochs=epochs)
    y_pred = log_reg.predict(X_test)

    print('y_test', y_test)
    print('y_pred', y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Accuracy: {accuracy:.4f}")