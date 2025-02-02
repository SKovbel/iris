import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from dataset import Iris

# K-Means: An unsupervised clustering algorithm that groups 
# similar data points into clusters based on their similarity.
class KMeanModel():
    def __init__(self, n_clusters=3, random_seed=1, n_init=10, max_iter=100, tol=1e-4):
        self.tol = tol
        self.n_init = n_init
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.random_seed = random_seed
    
    def forward(self, X):
        kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_seed, 
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol)
        clusters = kmeans.fit_predict(X)
        return clusters, kmeans.cluster_centers_

    def forward_(self, X):
        np.random.seed(self.random_seed)
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]  # Randomly centroids
        for _ in range(self.max_iter):
            # Step 1: Assign points to closest centroid, distance to centroids, label of closest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)  # 
            labels = np.argmin(distances, axis=1)
            # Step 2: Compute new centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            # Step 3: Check for convergence (if centroids stop changing)
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            centroids = new_centroids
            labels_ = labels 

        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)
        return clusters, centroids

    def visualize(self, title, X, y, names, clusters, centeroids, show=True):
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
        plt.scatter(centeroids[:, 0], centeroids[:, 1], c='yellow', marker='X', s=100, label='Centroids', edgecolors='k')
        plt.title(f"K-Means {title}")
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
    kmeans = KMeanModel(n_clusters=n_clusters, random_state=random_seed, n_init=n_init)

    # Data
    X, y = iris.array_dataset(one_hot_y=False, normilize=True)
    names = iris.speacias_names()

    # KMean lib
    clusters, centeroids = kmeans.forward(X)
    kmeans.visualize("Sklearn", X, y, names, clusters, centeroids, show=False)

    # Scratch
    clusters, centeroids = kmeans.forward_(X)
    kmeans.visualize("Scratch", X, y, names, clusters, centeroids)