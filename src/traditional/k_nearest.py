import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from dataset import Iris

class KNearestModel():
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsClassifier(n_neighbors=k_neighbors)

    def train(self, X_train, y_train):
        self.knn.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.knn.predict(X_test)
        return y_pred

if __name__ == '__main__':
    k_neighbors = 5  # Number of neighbors

    iris = Iris()
    knn = KNearestModel(k_neighbors)

    # data
    X_train, X_test, y_train, y_test = iris.numpy_dataset(test_size=0.2, one_hot_y=False, normilize=True)
    names = iris.speacias_names()

    # Test
    knn.train(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN Accuracy (k={k_neighbors}): {accuracy:.4f}")

    # Example Prediction
    sample_x = iris.normilize(np.array([[5.1, 3.5, 1.4, 0.2]]), fit=False)
    prediction = knn.predict(sample_x)
    print(prediction)
    print(f"Predicted Iris Class: {names[prediction[0]]}")
