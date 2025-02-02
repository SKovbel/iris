import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

from dataset import Iris

# Decision Trees: Tree-like models used for both classification and regression tasks, making decisions by following a set of rules based on input features.
class DTree():
    def __init__(self, depth=3, random_seed=1):
        self.random_seed = random_seed
        self.dtree = DecisionTreeClassifier(max_depth=depth, random_state=random_seed)

    def train(self, X_test, y_test):
        return self.dtree.fit(X_test, y_test)

    def predict(self, X_test):
        return self.dtree.predict(X_test)

    def accuracy(self, X_test, y_test):
        predictions = self.predict(X_test)
        return accuracy_score(y_test, predictions)

    def visualize(self, feature_names, class_names):
        plt.figure(figsize=(12, 8))
        plot_tree(self.dtree, feature_names=feature_names, class_names=class_names, filled=True)
        plt.show()

if __name__ == '__main__':
    random_seed=1

    iris = Iris()
    model = DTree()

    X_train, X_test, y_train, y_test = iris.numpy_dataset(test_size=0.2, one_hot_y=False, normilize=True)

    # test
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy_ = accuracy_score(y_test, y_pred)
    print(f"Desision tree Accuracy scratch: {accuracy_:.4f}")

    model.visualize(iris.feature_names(), iris.class_names())