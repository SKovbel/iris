import time
import numpy as np
from dataset import Iris

class NNMseSigmoid:
    biasses = []
    weights = []

    def __init__(self, dim=(3, 6, 4)):
        self.size = len(dim) - 1
        self.start_time = time.time()

        for i in range(self.size):
            self.weights.append(0.1*np.random.random((dim[i], dim[i + 1])) + 0.1)
            self.biasses.append(0.1*np.random.random((dim[i + 1])) + 0.1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, y):
        return y * (1 - y)

    def mse(self, Y, pred):
        return np.mean(np.square(pred - Y))

    def forward(self, X):
        outputs = [X]
        for i in range(self.size):
            X = np.dot(X, self.weights[i]) + self.biasses[i]
            X = self.sigmoid(X)
            outputs.append(X)
        return outputs

    def backward(self, X, Y, lr=0.1):
        outputs = self.forward(X)
        error = outputs[-1] - Y
        for i in range(self.size -1, -1, -1):
            grad = error * self.dsigmoid(outputs[i+1])
            self.biasses[i] -= lr * np.mean(grad, axis=0)
            self.weights[i] -= lr * np.dot(outputs[i].T, grad)
            error = np.dot(grad, self.weights[i].T) if i > 0 else None
        return self.mse(Y, outputs[-1])

    def predict(self, X):
        return self.forward(X)[-1]

    def train(self, X, Y, epochs=1000, lr=0.1):
        for e in range(epochs):
            loss = self.backward(X, Y, lr)
            print(f"Epoch = {e}, samples = {len(X)}, loss = {loss}, time = {round(time.time() - self.start_time, 3)}s.")


if __name__ == '__main__':
    iris = Iris()
    X_train, X_test, y_train, y_test = iris.numpy_dataset(test_size=0.2)
    model = NNMseSigmoid(dim=(4, 6, 3))
    model.train(X_train, y_train, epochs=1000, lr=0.005)
    y_pred = model.predict(X_test)
    iris.show_comparision(y_pred, y_test)