import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import Iris as Dataset

class IrisModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(3+4, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.nn(x)

    def train_model(self, X_train, y_train, epochs=100, lr=0.01):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.forward(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

def combinate_samples(X, Y):
    X_train, y_train = [], []
    def add_combination(x, y, v):
        X_train.append(np.concatenate((x, v)).tolist())
        y_train.append(y[v.index(max(v))].item() == 1)
    for x, y in zip(X, Y):
        add_combination(x, y, [0, 0, 1])
        add_combination(x, y, [0, 1, 0])
        add_combination(x, y, [1, 0, 0])
    return torch.tensor(X_train, dtype=torch.float32), \
           torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

if __name__ == '__main__':
    iris = Dataset()
    X, Y = iris.array_dataset(test_size=0.0, normilize=True)
    X_train, y_train = combinate_samples(X, Y)

    model = IrisModel()
    model.train_model(X_train, y_train, epochs=100, lr=0.01)

    y_var = [[0,0,1], [0,1,0], [1,0,0]]
    for x, y in zip(X, Y):
        for y_ in y_var:
            xt = torch.tensor(np.concatenate((x, y_)).tolist(), dtype=torch.float32)
            yt = model(xt)
            print(y, y_, round(yt.item(), 2), '%')
