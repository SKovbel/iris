import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from dataset import Iris

# SVD - singular Value Decompositions - Matrix fractorization technique for dimensionality reduction and latent features 
class SVD():
    def __init__(self, random_seed=1):
        self.random_seed = random_seed
        self.ada_boost = AdaBoostClassifier(n_estimators=50, random_state=random_seed)

    def predict(self, X, dim=2):
        U, sigma, Vt = np.linalg.svd(X)
        # Sigma is returned as a 1D array; convert it to a diagonal matrix
        Sigma_matrix = np.diag(sigma)
        # Step 3: Project data into the first two principal components (dimensions)
        return U[:, :dim] @ Sigma_matrix[:dim, :dim]

    def visualize(self, X_svd, y):
        plt.figure(figsize=(8, 5))
        plt.scatter(X_svd[:, 0], X_svd[:, 1], c=y, cmap='viridis', edgecolors='k', marker='o', s=100)
        plt.title('SVD of Iris Dataset')
        plt.xlabel('First Singular Vector')
        plt.ylabel('Second Singular Vector')
        plt.colorbar(label='Species')
        plt.show()

if __name__ == '__main__':
    random_seed=1
    n_clusters=3
    n_init=10

    iris = Iris()
    svd = SVD()

    X, y = iris.array_dataset(one_hot_y=False, normilize=True)

    # test
    X_svd = svd.predict(X)
    svd.visualize(X_svd, y)