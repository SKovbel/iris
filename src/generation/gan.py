import torch
import torch.nn as nn
import torch.optim as optim

from dataset import Iris
from stats import Stats
from simple.torch_softmax import NNSoftmax

# @todo bad result
class Generator(nn.Module):
    def __init__(self, latent_dim=8, hidden_size=64, output_size=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_size=4, hidden_size=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class GAN:
    def __init__(self, input_size=4, hidden_size=64, latent_dim=8):
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size

        self.generator = Generator(latent_dim, hidden_size, output_size=input_size)
        self.discriminator = Discriminator(input_size=input_size, hidden_size=hidden_size)

    def train(self, dataloader, batch_size=16, epochs=100, lr=0.0002):
        criterion = nn.BCELoss()
        optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        for epoch in range(epochs):
            for real_data, in dataloader:
                batch_size = real_data.size(0)
                real_labels = torch.ones(batch_size, 1)
                fake_labels = torch.zeros(batch_size, 1)

                # Pass real data to discriminator
                optimizer_D.zero_grad()
                real_preds = self.discriminator.forward(real_data)
                real_loss = criterion(real_preds, real_labels)

                # Generate and pass fake data to discriminator
                z = torch.randn(batch_size, self.latent_dim)
                fake_data = self.generator.forward(z)
                fake_preds = self.discriminator.forward(fake_data.detach())
                fake_loss = criterion(fake_preds, fake_labels)

                # Train discriminator
                loss_D = real_loss + fake_loss
                loss_D.backward()
                optimizer_D.step()

                # Fool discriminator
                optimizer_G.zero_grad()
                fake_preds = self.discriminator.forward(fake_data)
                loss_G = criterion(fake_preds, real_labels)

                # Train generator
                loss_G.backward()
                optimizer_G.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

    def samples(self, num_samples=150):
        z = torch.randn(num_samples, self.latent_dim)
        samples = self.generator(z).detach().numpy()
        return samples

if __name__ == '__main__':
    lr=0.001
    batch_size = 32
    input_dim = 4
    hidden_dim = 16
    latent_dim = 8
    epochs = 1000
    samples_count = 150

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    iris = Iris()
    stats = Stats()
    softmax = NNSoftmax()
    gan = GAN( input_size=4, hidden_size=64, latent_dim=8)
    
    # train softmax on real data
    X_train, X_test, Y_train, Y_test = iris.numpy_dataset(test_size=0.2, normilize=True)
    softmax.backward(X_train, Y_train, epochs=2000, lr=0.1)

    # train
    dataloader = iris.torch_dataset(batch_size=batch_size, normilize=True)
    gan.train(dataloader, batch_size, epochs, lr)
    samples_X = gan.samples(samples_count)

    # classify new samples with softmax
    samples_y = softmax.predict(samples_X)
    print("\nGenerated Samples:")
    iris.show_probability(iris.denormilize(samples_X), samples_y)
    stats.sns_pairplot(iris.denormilize(samples_X), samples_y, show_default=True)

