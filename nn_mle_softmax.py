import time
import numpy as np
from dataset import Iris as Dataset

class NNet:
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

    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def dsoftmax(self, y):
        jacobians = []
        for i in range(len(y)):
            z = y[i].reshape(-1, 1)
            jacobian = np.diagflat(z) - np.dot(z, z.T)
            jacobians.append(jacobian)
        return np.array(jacobians)

    def mle(self, y, pred):
        return -np.mean(np.sum(y * np.log(pred + 1e-9), axis=1))

    def forward(self, X):
        outputs = [X]
        for i in range(self.size):
            X = np.dot(X, self.weights[i]) + self.biasses[i]
            if i == self.size - 1:
                X = self.softmax(X)
            else:
                X = self.sigmoid(X)
            outputs.append(X)
        return outputs

    def backward(self, X, Y, lr=0.1):
        outputs = self.forward(X)
        error = outputs[-1] - Y
        loss = self.mle(Y, outputs[-1])

        for i in range(self.size -1, -1, -1):
            if i == self.size - 1:
                jacobian = self.dsoftmax(outputs[-1])
                grad = np.einsum('ijk,ik->ij', jacobian, error)
            else:
                grad = error * self.dsigmoid(outputs[i+1])

            self.biasses[i] -= lr * np.mean(grad, axis=0)
            self.weights[i] -= lr * np.dot(outputs[i].T, grad)
            error = np.dot(grad, self.weights[i].T) if i > 0 else None
        return loss

    def predict(self, X):
        return self.forward(X)[-1]

    def train(self, X, Y, epochs=1000, lr=0.1):
        for e in range(epochs):
            loss = self.backward(X, Y, lr)
            print(f"Epoch = {e}, samples = {len(X)}, loss = {loss}, time = {round(time.time() - self.start_time, 3)}s.")


if __name__ == '__main__':
    ds = Dataset(test_size=0.2)
    X_train, X_test, y_train, y_test = ds.train_data()
    model = NNet(dim=(4, 6, 3))
    model.train(X_train, y_train, epochs=500, lr=0.005)
    y_pred = model.predict(X_test)
    ds.compare_pred(y_pred, y_test)
