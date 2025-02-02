import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from dataset import Iris

# Support Vector Machines (SVM): Used for classification tasks, 
# SVM tries to find the optimal hyperplane that best separates data points belonging to different classes.
class SVMModel():
    def __init__(self, random_seed=1):
        self.random_seed = random_seed
        self.svm = SVC(kernel='linear', random_state=self.random_seed)

    def train(self, X_train, y_train):
        self.svm.fit(X_train, y_train)

    def predict(self, X_test):
        return self.svm.predict(X_test)

# @todo bad result
class SVMModel_():
    def __init__(self, n_features, n_classes):
        self.n_classes = n_classes
        self.n_features = n_features
        self.biases = np.zeros(n_classes)
        self.weights = np.zeros((n_classes, n_features))

    def train(self, X, y, epochs=100, lr=0.001, lambda_reg=0.01):
        for _ in range(epochs):
            for i, x in enumerate(X):
                true_class = y[i]
                for cls in range(self.n_classes):
                    margin = np.dot(self.weights[cls], x) + self.biases[cls] - (np.dot(self.weights[true_class], x) + self.biases[true_class])

                    if cls == true_class:
                        if margin > -1:
                            # Update weights and biases for the true class
                            self.weights[cls] -= lr * (lambda_reg * self.weights[cls] + x)
                            self.biases[cls] -= lr
                    else:
                        if margin < 1:
                            # Update weights and biases for the incorrect class
                            self.weights[cls] -= lr * (lambda_reg * self.weights[cls] - x)
                            self.biases[cls] -= lr


    def predict(self, X):
        scores = np.dot(X, self.weights.T) + self.biases
        return np.argmax(scores, axis=1)

if __name__ == '__main__':
    lr = 0.0001
    lambda_reg = 0.01
    epochs = 1000

    iris = Iris()

    # data
    X_train, X_test, y_train, y_test = iris.numpy_dataset(test_size=0.2, one_hot_y=False, normilize=True)

    # Test
    svm = SVMModel()
    svm.train(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Accuracy: {accuracy:.4f}")

    # Scratch
    svm_ = SVMModel_(n_features=4, n_classes=3)
    svm_.train(X_train, y_train, epochs=epochs, lr=lr, lambda_reg=lambda_reg)
    y_pred_ = svm_.predict(X_test)
    accuracy_ = accuracy_score(y_test, y_pred_)
    print(f"SVM Accuracy scratch: {accuracy_:.4f}")
