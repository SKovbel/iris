import numpy as np
from dataset import Iris as Dataset

# Monte Carlo
class BayesNNet:
    layers = []

    def __init__(self, layers):
        for layer in layers:
            self.layers.append({
                'func': layer['func'],
                'w_mean': np.random.randn(layer['in'], layer['out']),
                'w_std': np.abs(np.random.randn(layer['in'], layer['out'])),
                'b_mean': np.random.randn(layer['out']),
                'b_std': np.abs(np.random.randn(layer['out'])),
            })

    def softmax(self, X):
        exp_x = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def relu(self, X):
        return np.maximum(0, X)

    def drelu(self, Y):
        return Y > 0

    def dsoftmax(self, Y):
        jacobians = []
        for i in range(len(Y)):
            Z = Y[i].reshape(-1, 1)
            jacobian = np.diagflat(Z) - np.dot(Z, Z.T)
            jacobians.append(jacobian)
        return np.array(jacobians)

    def log_likelihood(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

    def sample(self):
        return [{
            'f': layer['func'],
            'w': np.random.normal(layer['w_mean'], layer['w_std']),
            'b': np.random.normal(layer['b_mean'], layer['b_std']),
        } for layer in self.layers]

    def forward(self, X):
        predicts = [X]
        for layer in self.sample():
            y = np.dot(X, layer['w']) + layer['b']
            X = self.relu(y) if layer['f'] == 'relu' else self.softmax(y)
            predicts.append(X)
        return predicts

    def backward(self, X, y, samples=100, lr=0.01):
        loss = 0.0
        for _ in range(samples):
            preds = self.forward(X)
            grads = preds[-1] - y
            loss += self.log_likelihood(y, preds[-1])

            for i in range(len(self.layers) -1, -1, -1):
                if self.layers[i]['func'] == 'relu':
                    grads = grads * self.drelu(preds[i+1])

                next_grads = np.dot(grads, self.layers[i]['w_mean'].T)
                self.layers[i]['w_mean'] -= lr * np.dot(preds[i].T, grads)
                self.layers[i]['b_mean'] -= lr * np.sum(grads, axis=0)
                grads = next_grads

        return loss

    def train(self, X, y, epochs=100, samples=10, lr=0.005):
        for e in range(epochs):
            loss = self.backward(X, y, samples, lr)
            print(f"Epoch {e+1}/{epochs}, Loss: {loss/samples:.4f}")

    def predict(self, X, samples=100):
        predictions = []
        for _ in range(samples):
            predictions.append(self.forward(X)[-1])
        return np.array(predictions).mean(axis=0)

if __name__ == '__main__':
    bnn = BayesNNet(layers=(
        {'func': 'relu',    'in': 4, 'out': 6},
        {'func': 'softmax', 'in': 6, 'out': 3}
    ))

    ds = Dataset()

    X_train, X_test, y_train, y_test = ds.train_norm()
    bnn.train(X_train, y_train, epochs=100, samples=10, lr=0.01)

    y_pred = bnn.predict(X_test)
    ds.compare_pred(y_pred, y_test)
