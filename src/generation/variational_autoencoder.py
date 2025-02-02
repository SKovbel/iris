import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Iris
from stats import Stats
from neural.torch_softmax import NNSoftmax

# VAE Model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    # PDF curve bell
    # CDF - cummulative of bell from -∞ to +∞
    # eps = random variable N(0,1), mean=0 std=1 or 34%
    # first layer relu
    # second layer is mu and logvar
    # output z = mu + eps * std 
    #          = mu + eps * exp(1/2 * logvar)
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # relu(z)
    def decode(self, z):
        h = F.relu(self.fc2(z))
        return self.fc3(h)  # Reconstructed features

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar

    def train_model(self, data, epochs, learning_rate, device):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            self.train()
            train_loss = 0
            for batch in data:
                x = batch[0].to(device)
                optimizer.zero_grad()
                recon_x, mu, logvar = self.forward(x)
                loss = self.loss_function(recon_x, x, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {train_loss / len(data):.4f}")

    # KL Divergence (KLD) ensures that the learned latent space distribution is close to a standard normal distribution (N(0,1)).
    # DKL(q(z∣x) ∣∣ p(z)) = −1/2 * ∑(1 + log(σ^2) − μ^2 − σ^2)
    @staticmethod
    def loss_function(recon_X, X, mu, logvar):
        # Mean Squared Error, Reconstruction Loss
        # KL Divergence, Regularization
        MSE = F.mse_loss(recon_X, X, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD

    @staticmethod
    def loss_function_todo(recon_X, X, mu, logvar):
        # Binary Cross Entropy, Reconstruction Loss
        # KL Divergence, Regularization
        BCE = F.binary_cross_entropy(recon_X, X, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def samples(self):
        self.eval()
        samples = []
        with torch.no_grad():
            z = torch.randn(samples_count, latent_dim).to(device)
            sample = self.decode(z).cpu().numpy()
            samples.append(sample.tolist())
        return samples[0]

if __name__ == '__main__':
    # Parameters
    input_dim = 4
    hidden_dim = 16
    latent_dim = 32
    batch_size = 16
    learning_rate = 1e-3
    epochs = 100
    samples_count = 150
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    iris = Iris()
    stats = Stats()
    softmax = NNSoftmax()
    vae = VAE(input_dim, hidden_dim, latent_dim).to(device)

    # train 
    dataloader = iris.torch_dataset(batch_size=batch_size, normilize=True)
    vae.train_model(dataloader, epochs, learning_rate, device)
    samples_X = vae.samples()

    # train softmax on real data
    X_train, X_test, Y_train, Y_test = iris.numpy_dataset(test_size=0.2, normilize=True)
    softmax.backward(X_train, Y_train, epochs=2000, lr=0.1)

    # classify new samples with softmax
    samples_y = softmax.predict(samples_X)
    print("\nGenerated Samples:")
    iris.show_probability(iris.denormilize(samples_X), samples_y)
    stats.sns_pairplot(iris.denormilize(samples_X), samples_y, show_default=True)