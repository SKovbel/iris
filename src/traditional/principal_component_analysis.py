import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from dataset import Iris

# Principal Component Analysis (PCA): 
# A dimensionality reduction technique used to reduce the number of features in a dataset while preserving its variance.
class PCAModel:
    def predict(self, X, n_components=2):
        pca = PCA(n_components=n_components)
        return pca.fit_transform(X)

    def predict_(self, X, n_components=2):
        # Step 1: Compute the covariance matrix
        cov_matrix = np.cov(X.T)

        # Step 3: Calculate the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Step 4: Sort eigenvalues and eigenvectors in decreasing order
        sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort in descending order
        eigenvectors_sorted = eigenvectors[:, sorted_indices]
        # eigenvalues_sorted = eigenvalues[sorted_indices]
        # variance = eigenvalues_sorted / np.sum(eigenvalues_sorted)
        # Step 5: Project the data onto the first two principal components
        X_pca = X.dot(eigenvectors_sorted[:, :n_components])  # Project the data to 2D

        return X_pca

    def visualize(self, label, X, show=True):
        plt.figure(figsize=(8, 5))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', marker='o', s=100)
        plt.title(f'PCA {label}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(label='Species')
        if show:
            plt.show()

if __name__ == '__main__':
    random_seed=1
    n_clusters=3
    n_init=10

    iris = Iris()
    pca = PCAModel()

    X, y = iris.array_dataset(one_hot_y=False, normilize=True)
    names = iris.class_names()

    X_pca1 = pca.predict(X, n_components=2)
    X_pca2 = pca.predict_(X, n_components=2)

    print(X_pca2)
    pca.visualize("SKI-1", X_pca1, show=False)
    pca.visualize("SKI-2", X_pca2)
