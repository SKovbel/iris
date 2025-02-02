import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataset import Iris as Dataset

class NNSoftmax(nn.Module):
    def __init__(self):
        super(NNSoftmax, self).__init__()
        self.start_time = time.time()
        self.layer_in = nn.Linear(4, 6)
        self.layer_out = nn.Linear(6, 3)

    def forward(self, X):
        X = F.sigmoid(self.layer_in(X))
        X = F.softmax(self.layer_out(X), dim=1)
        return X

    def backward(self, X, Y, epochs=1000, lr=0.1):
        inputs = torch.tensor(X, dtype=torch.float32)
        targets = torch.tensor(Y, dtype=torch.float32)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=lr)

        for e in range(epochs):
            outputs = self.forward(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch = {e}, samples = {len(X)}, loss = {loss.item()}, time = {round(time.time() - self.start_time, 3)}s.")

    def predict(self, X):
        inputs = torch.tensor(X, dtype=torch.float32)
        outputs = self.forward(inputs)
        print(outputs)
        return outputs.detach().numpy()

if __name__ == '__main__':
    iris = Dataset()
    X_train, X_test, Y_train, Y_test = iris.numpy_dataset(test_size=0.2)

    model = NNSoftmax()
    model.backward(X_train, Y_train, epochs=2000, lr=0.1)
    Y_pred = model.predict(X_test)
    iris.show_comparision(Y_pred, Y_test)
    #iris.show_probability(Y_pred)
