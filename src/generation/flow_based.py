import random
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import Iris
from stats import Stats
from neural.torch_softmax import NNSoftmax

class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(AffineCouplingLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Define a simple feed-forward neural network for scaling and shifting
        self.nn = nn.Sequential(
            nn.Linear(input_dim // 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, input_dim // 2),
        )

    def forward(self, x, logdet=0.0, reverse=False):
        x1, x2 = torch.chunk(x, 2, dim=-1) # Split input into two halves
        if not reverse:
            shift, scale = self.nn(x1).chunk(2, dim=-1)
            z = x2 * torch.exp(scale) + shift
            logdet = logdet + torch.sum(scale, dim=-1)  # Jacobian determinant (log-sum of scaling factors)
            return torch.cat([x1, z], dim=-1), logdet
        else:
            shift, scale = self.nn(x1).chunk(2, dim=-1)
            x2 = (x2 - shift) * torch.exp(-scale)
            return torch.cat([x1, x2], dim=-1), logdet

class FlowBasedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_layers=2):
        super(FlowBasedModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.layers = nn.ModuleList([AffineCouplingLayer(input_dim, hidden_dim) for _ in range(n_layers)])

    def forward(self, x, logdet=0.0, reverse=False):
        for layer in self.layers:
            x, logdet = layer(x, logdet, reverse=reverse)
        return x, logdet

    def likelihood(self, x):
        z, logdet = self.forward(x)
        log_likelihood = -0.5 * torch.sum(z ** 2, dim=-1) - logdet
        return log_likelihood

    def train_model(self, dataloader, epochs=500, lr=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            self.train()  # Set model to training mode
            total_loss = 0
            for data_batch, in dataloader:  # Extract tensor from DataLoader
                optimizer.zero_grad()

                loss = -self.likelihood(data_batch).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 20 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {total_loss / len(dataloader):.4f}')

    def generate(self, num_samples=150):
        z = torch.randn((num_samples, self.input_dim))
        x, logdet = self.forward(z, reverse=True)
        return x.detach().numpy()

if __name__ == "__main__":
    lr = 1e-4
    epochs = 200
    batch_size = 16
    input_dim=4
    hidden_dim=16
    n_layers=2

    iris = Iris()
    stats = Stats()
    softmax = NNSoftmax()

    # Train the model
    dataloader = iris.torch_dataset(batch_size=batch_size, normilize=True)
    flow = FlowBasedModel(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    flow.train_model(dataloader, epochs=epochs, lr=lr)
    samples_X = flow.generate(num_samples=150)

    # train softmax on real data
    X_train, X_test, Y_train, Y_test = iris.numpy_dataset(test_size=0.2, normilize=True)
    softmax.backward(X_train, Y_train, epochs=2000, lr=0.1)

    # classify new samples with softmax
    samples_y = softmax.predict(samples_X)
    print("\nGenerated Samples:")
    iris.show_probability(iris.denormilize(samples_X), samples_y)
    stats.sns_pairplot(iris.denormilize(samples_X), samples_y, show_default=True)
