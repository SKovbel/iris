import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from dataset import Iris

# Gaussian Mixture Model - representing data as mixture of Gaussian distibutions
class GMMModel():
    def __init__(self, n_components=3, random_seed=1):
        self.random_seed = random_seed
        self. gmm = GaussianMixture(n_components=n_components, random_state=random_seed)

    def train(self, X_train):
        self.gmm.fit(X_train)

    def predict(self, X_test):
        y_pred = self.gmm.predict(X_test)
        return y_pred

    def info(self):
        print("\nGMM components (mean, covariance, and weight):")
        for i in range(self.gmm.n_components):
            print(f"Component {i+1}:")
            print(f"  Mean: {self.gmm.means_[i]}")
            print(f"  Covariance: \n{self.gmm.covariances_[i]}")
            print(f"  Weight: {self.gmm.weights_[i]}\n")

    def visualize(self, X, y_pred):
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', marker='o', edgecolor='k', s=100)
        plt.title('Clustering using Gaussian Mixture Model (GMM) and Expectation-Maximization (EM)')
        plt.xlabel('Sepal Length')
        plt.ylabel('Sepal Width')
        plt.colorbar(label='Cluster')
        plt.show()

if __name__ == '__main__':
    lr = 0.0001
    lambda_reg = 0.01
    epochs = 1000

    iris = Iris()
    gmm = GMMModel()

    X, y = iris.array_dataset()
    y_pred = gmm.train(X)
    y_pred = gmm.predict(X)

    gmm.info()
    gmm.visualize(X, y_pred)
