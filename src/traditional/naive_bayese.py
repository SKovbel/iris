import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from dataset import Iris

# Naive Bayes: A probabilistic classifier based on Bayes' theorem, often used for text classification and spam filtering.
class GaussianNBModel():
    def __init__(self, random_seed=1):
        self.gnb = GaussianNB()

    def train(self, X_train, y_train):
        self.gnb.fit(X_train, y_train)

    def predict(self, X_test):
        return  self.gnb.predict(X_test)

# hand made
class GaussianNBModel_:
    def __init__(self):
        self.class_probabilities = {}
        self.mean = {}
        self.var = {}

    def train(self, X, y):
        n_samples, n_features = X.shape
        classes = np.unique(y)
        # Calculate class probabilities
        for c in classes:
            class_data = X[y == c]
            self.class_probabilities[c] = len(class_data) / n_samples
            # Calculate mean and variance for each feature per class
            self.mean[c] = np.mean(class_data, axis=0)
            self.var[c] = np.var(class_data, axis=0)
    
    def gaussian_pdf(self, x, mean, var):
        exponent = np.exp(- (x - mean) ** 2 / (2 * var))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent

    def predict(self, X):
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)
    
    def _predict_single(self, x):
        class_probabilities = {}
        for c in self.class_probabilities:
            # Log of prior probability
            prior = np.log(self.class_probabilities[c])
            # Calculate likelihood using the Gaussian PDF for each feature
            likelihood = np.sum(np.log(self.gaussian_pdf(x, self.mean[c], self.var[c])))
            # Combine prior and likelihood
            class_probabilities[c] = prior + likelihood
        # Return the class with the highest posterior probability
        return max(class_probabilities, key=class_probabilities.get)

if __name__ == '__main__':
    random_seed=1
    n_clusters=3
    n_init=10

    iris = Iris()
    gnb = GaussianNBModel()
    gnb_ = GaussianNBModel_()

    X_train, X_test, y_train, y_test = iris.numpy_dataset(test_size=0.2, one_hot_y=False, normilize=True)

    # test
    gnb.train(X_train, y_train)
    y_pred = gnb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    gnb_.train(X_train, y_train)
    y_pred_ = gnb_.predict(X_test)
    accuracy_ = accuracy_score(y_test, y_pred)

    print(f"Naive Bayes Accuracy: {accuracy:.4f}")
    print(f"Naive Bayes Accuracy_: {accuracy_:.4f}")
