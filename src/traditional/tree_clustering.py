import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

from dataset import Iris

# Builds tree cluster based on data similarities and
class HCModel():
    def __init__(self, random_seed=1):
        self.random_seed = random_seed
        self.model = AgglomerativeClustering(n_clusters=3, linkage='ward')

    def predict(self, X_test):
        y_pred = self.model.fit_predict(X_test)
        return y_pred

    def visualize(self, title, X, y, names, clusters, show=True):
        species_colors = ['red', 'blue', 'green']
        species_markers = ['o', 's', 'D']

        plt.figure(figsize=(8, 5))
        plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.7, edgecolors='k')
        for i in range(3):  # Loop through each species
            plt.scatter(
                X[(y == i), 0], X[(y == i), 1], 
                label=names[i], 
                color=species_colors[i], 
                marker=species_markers[i], 
                alpha=0.7, edgecolors='k')
        plt.title(f"Cluster {title}")
        plt.xlabel("Sepal Length (normalized)")
        plt.ylabel("Sepal Width (normalized)")
        plt.legend()
        if show:
            plt.show()

if __name__ == '__main__':
    random_seed=1
    n_clusters=3
    n_init=10

    iris = Iris()
    hc = HCModel()

    X, y = iris.array_dataset(one_hot_y=False, normilize=True)
    names = iris.class_names()

    # test
    clusters = hc.predict(X)
    hc.visualize("Sklearn", X, y, names, clusters)
