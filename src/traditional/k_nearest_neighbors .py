import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from dataset import Iris

# K-Nearest Neighbors (KNN): A simple instance-based learning algorithm used for classification and regression tasks,
# where predictions are based on the majority class of the k-nearest data points.
class KNearestModel():
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsClassifier(n_neighbors=k_neighbors)

    def train(self, X_train, y_train):
        self.knn.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.knn.predict(X_test)
        return y_pred

    def train_(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict_(self, X_test):
        def predict__(x):
                # Step 1: Compute distances from x to all points in the training set
                distances = [np.linalg.norm(x_train - x) for x_train in self.X_train]
                # Step 2: Sort distances and get the indices of the k-nearest neighbors
                k_indices = np.argsort(distances)[:self.n_neighbors]
                # Step 3: Get the labels of the k-nearest neighbors
                k_nearest_labels = [self.y_train[i] for i in k_indices]
                # Step 4: Return the most common class label
                most_common = Counter(k_nearest_labels).most_common(1)
                return most_common[0][0]
        predictions = [predict__(x) for x in X_test]
        return np.array(predictions)

if __name__ == '__main__':
    k_neighbors = 21

    iris = Iris()
    knn = KNearestModel(k_neighbors)

    # data
    X_train, X_test, y_train, y_test = iris.numpy_dataset(test_size=0.2, one_hot_y=False, normilize=True)
    names = iris.class_names()

    # Test
    knn.train(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN Accuracy (k={k_neighbors}): {accuracy:.4f}")

    # Test scratch
    knn.train_(X_train, y_train)
    y_pred = knn.predict_(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN Accuracy scratch (k={k_neighbors}): {accuracy:.4f}")

    # Example Prediction
    sample_x = iris.normilize(np.array([[5.1, 3.5, 1.4, 0.2]]), fit=False)
    prediction = knn.predict(sample_x)
    print(prediction)
    print(f"Predicted Iris Class: {names[prediction[0]]}")
