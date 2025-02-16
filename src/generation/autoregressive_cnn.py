import torch
import torch.nn as nn
import torch.optim as optim

from dataset import Iris
from stats import Stats
from neural.classification import NNSoftmax

class AutoRegCnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size=16, epochs=200, lr=0.001):
        super(AutoRegCnn, self).__init__()

        self.lr = lr
        self.epochs = epochs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64 * input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

        self.relu = nn.ReLU()

    def forward(self, X):
        X = X.unsqueeze(1)
        X = self.relu(self.conv1(X))
        X = self.relu(self.conv2(X))
        X = self.relu(self.conv3(X))
        X = X.view(X.size(0), -1)  # Flatten
        X = self.relu(self.fc1(X))
        X = self.fc2(X)
        return X

    def train_model(self, data, epochs=100):
        loss_func = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for inputs_batch, in data:
                optimizer.zero_grad()
                output = self.forward(inputs_batch)
                loss = loss_func(output, inputs_batch)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

            if epoch % 20 == 0:
                print(f'Epoch {epoch}/{self.epochs}, Loss: {epoch_loss / len(data)}')

    def generate(self, examples):
        self.eval()
        samples = []
        with torch.no_grad():
            for batch, in examples:
                output = self.forward(batch)
                samples.extend(output.cpu().numpy().tolist())
        return samples

if __name__ == '__main__':
    input_size = 4
    hidden_size = 64
    output_size = 4
    batch_size = 16
    epochs = 100
    num_samples = 150

    iris = Iris()
    stats = Stats()
    softmax = NNSoftmax()
    auto_reg = AutoRegCnn(input_size, hidden_size, output_size, batch_size)

    # train auto reg model
    dataloader = iris.torch_dataset(batch_size, normilize=True)
    auto_reg.train_model(dataloader, epochs)
    samples_X = auto_reg.generate(examples=dataloader)

    # train softmax on real data
    X_train, X_test, Y_train, Y_test = iris.numpy_dataset(test_size=0.2, normilize=True)
    softmax.backward(X_train, Y_train, epochs=2000, lr=0.1)

    # classify new samples with softmax
    print(samples_X)
    samples_y = softmax.predict(samples_X)
    print("\nGenerated Samples:")
    iris.show_probability(iris.denormilize(samples_X), samples_y)
    stats.sns_pairplot(iris.denormilize(samples_X), samples_y, show_default=True)
