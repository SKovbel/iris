import torch
import torch.nn as nn
import torch.optim as optim

from dataset import Iris
from stats import Stats
from xtorch.nn_softmax import NNSoftmax

class AutoRegLstm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size=16, epochs=200, lr=0.001):
        super(AutoRegLstm, self).__init__()

        self.lr = lr
        self.epochs = epochs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        X = X.unsqueeze(1)
        lstm_out, (hn, cn) = self.rnn(X)
        out = self.fc(lstm_out[:, -1, :])  # last time step, [Batch, Time Steps, Hidden Size]
        return out

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

    def samples(self, examples):
        self.eval()
        samples = []
        with torch.no_grad():
            for batch, in examples:
                output = self.forward(batch)
                samples.extend(output.cpu().numpy().tolist())
        return samples

    def samples2(self, seed_feature, num_samples=50):
        self.eval()
        samples = []
        with torch.no_grad():
            input = seed_feature.unsqueeze(0)
            for _ in range(num_samples):
                output = self.forward(input)
                samples.append(output.squeeze(0).cpu().numpy().tolist())
                input = output
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
    auto_reg = AutoRegLstm(input_size, hidden_size, output_size, batch_size)

    # train auto reg model
    dataloader = iris.torch_dataset(batch_size, normilize=True)
    auto_reg.train_model(dataloader, epochs)
    samples_X = auto_reg.samples(examples=dataloader)

    seed_feature, = next(iter(dataloader))
    seed_feature = seed_feature[0]
    samples_X_ = auto_reg.samples2(seed_feature, num_samples=num_samples)

    # train softmax on real data
    X_train, X_test, Y_train, Y_test = iris.numpy_dataset(test_size=0.2, normilize=True)
    softmax.backward(X_train, Y_train, epochs=2000, lr=0.1)

    # classify new samples with softmax
    samples_y = softmax.predict(samples_X)
    print("\nGenerated Samples:")
    iris.show_probability(iris.denormilize(samples_X), samples_y)
    stats.sns_pairplot(iris.denormilize(samples_X), samples_y, show_default=True)
