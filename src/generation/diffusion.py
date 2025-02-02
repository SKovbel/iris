import torch
import torch.nn as nn
import torch.optim as optim

from dataset import Iris
from stats import Stats
from simple.torch_softmax import NNSoftmax

class Diffusion(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_steps=1000):
        super(Diffusion, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def add_noise(self, x, t, beta=0.1):
        noise = torch.randn_like(x)
        return torch.sqrt(1 - beta * t) * x + torch.sqrt(beta * t) * noise

    def denoise(self, x_t, t, beta=0.1):
        predicted_noise = self(x_t)
        t = t.view(-1, 1)  # Ensures t is of shape [batch_size, 1]
        sqrt_beta_t = torch.sqrt(beta * t)
        sqrt_1_minus_beta_t = torch.sqrt(1 - beta * t)

        return (x_t - sqrt_beta_t * predicted_noise) / sqrt_1_minus_beta_t

    def train_model(self, data, epochs=10, lr=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            for batch in data:
                if len(batch) < batch_size:
                    continue

                # Random time steps for forward diffusion process
                t = torch.randint(0, self.num_steps, (batch.size(0),), dtype=torch.long)
                t = t.float() / self.num_steps  # Normalize to [0, 1]

                x_t = self.add_noise(batch, t)
                x_reconstructed = self.denoise(x_t, t)

                # Compute the loss - difference between reconstructed and original data
                loss = loss_fn(x_reconstructed, batch)
                total_loss += loss.item()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data)}")

    def samples(self, num_samples=150):
        samples = torch.randn(num_samples, self.input_dim)
        for t in reversed(range(self.num_steps)):
            samples = self.denoise(samples, torch.tensor([t] * num_samples))
        return samples.detach().cpu().numpy()

if __name__ == "__main__":
    batch_size = 16
    num_steps = 100
    num_samples = 150
    epochs = 100

    iris = Iris()
    stats = Stats()
    softmax = NNSoftmax()
    diffusion = Diffusion(input_dim=4, hidden_dim=64, num_steps=10)

    # train softmax on real data
    X_train, X_test, Y_train, Y_test = iris.numpy_dataset(test_size=0.2, normilize=True)
    softmax.backward(X_train, Y_train, epochs=2000, lr=0.1)

    # train diffusion
    dataloader = iris.torch_dataset(batch_size=batch_size, normilize=True)
    diffusion.train_model(dataloader, epochs=epochs)
    samples_X = diffusion.samples(num_samples=num_samples)

    # classify new samples with softmax
    samples_y = softmax.predict(samples_X)
    print("\nGenerated Samples:")
    iris.show_probability(iris.denormilize(samples_X), samples_y)
    stats.sns_pairplot(iris.denormilize(samples_X), samples_y, show_default=True)
