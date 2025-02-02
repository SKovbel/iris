from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from dataset import Iris

# Random Forest: An ensemble method that combines multiple decision trees.
class RForest():
    def __init__(self, depth=3, n_estimators=100, random_seed=1):
        self.random_seed = random_seed
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=depth, random_state=random_seed
        )

    def train(self, X_train, y_train):
        return self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def accuracy(self, X_test, y_test):
        predictions = self.predict(X_test)
        return accuracy_score(y_test, predictions)

if __name__ == '__main__':
    random_seed = 1

    iris = Iris()
    model = RForest(depth=5, n_estimators=100) 

    X_train, X_test, y_train, y_test = iris.numpy_dataset(test_size=0.2, one_hot_y=False, normilize=True)

    model.train(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy_ = model.accuracy(X_test, y_test)
    print(f"Random Forest Accuracy: {accuracy_:.4f}")
