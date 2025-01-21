import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(1)

class Iris:
    def __init__(self, test_size=0.2):
        self.test_size=test_size
        self.scaler = StandardScaler()

    def load_data(self):
        data = load_iris()
        N = np.max(data.target) + 1
        self.X = data.data
        self.y = np.eye(N)[data.target]
        return self.X, self.y

    def train_data(self):
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=1)
        return X_train, X_test, y_train, y_test

    def train_norm(self):
        X_train, X_test, y_train, y_test = self.train_data()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def compare_pred(self, pred, y_test):
        norm = np.zeros_like(pred)
        norm[np.arange(pred.shape[0]), np.argmax(pred, axis=1)] = 1
        equal = 0
        for a, b in zip(y_test, norm):
            equal += (a == b).all()
            print(a, b)
        print(f"Accuracy = {int(100*equal/len(y_test))}%")
