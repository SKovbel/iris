from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

from dataset import Iris

class Adaboost():
    def __init__(self, random_seed=1):
        self.random_seed = random_seed
        self.ada_boost = AdaBoostClassifier(n_estimators=50, random_state=random_seed)

    def train(self, X_test, y_test):
        return self.ada_boost.fit(X_test, y_test)

    def predict(self, X_test):
        return self.ada_boost.predict(X_test)

if __name__ == '__main__':
    random_seed=1
    n_clusters=3
    n_init=10

    iris = Iris()
    ada = Adaboost()

    X_train, X_test, y_train, y_test = iris.numpy_dataset(test_size=0.2, one_hot_y=False, normilize=True)

    # test
    ada.train(X_train, y_train)
    y_pred = ada.predict(X_test)

    accuracy_ = accuracy_score(y_test, y_pred)
    print(f"ADABoost Accuracy scratch: {accuracy_:.4f}")
