import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import Iris

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IrisEnv:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.index = 0

    def reset(self):
        self.index = 0
        return self.X[self.index]

    def step(self, action):
        if self.index >= len(self.X) - 1:
            self.index = 0

        correct = torch.argmax(action[0]) == torch.argmax(self.y[self.index])
        reward = 1 if correct else -1
        self.index += 1
        done = self.index >= len(self.y)
        next_state = self.X[min(self.index, len(self.y) - 1)]
        #print('a', action, 'y', self.y[self.index], 's', next_state, 'r', reward, 'd', done, 'corr', correct)
        #      'mx1', torch.argmax(action[0]).item(),
        #      'mx2', torch.argmax(self.y[self.index]).item())
        return next_state, reward, done

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        self.actor = nn.Linear(64, output_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, X):
        x = torch.sigmoid(self.fc1(X))
        x = torch.sigmoid(self.fc2(x))

        state_value = self.critic(x)
        action_value = self.actor(x)
        action_prob = torch.softmax(action_value, dim=-1)

        return action_prob, state_value

class PPO:
    def __init__(self, input_dim, output_dim, lr=0.001, gamma=0.99, epsilon=0.2, epochs=10):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.policy = ActorCritic(input_dim, output_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

    def compute_advantages(self, rewards, values, dones):
        returns = []
        discounted_reward = 0

        for i in reversed(range(len(rewards))):
            if dones[i]: 
                discounted_reward = 0
            discounted_reward = rewards[i] + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        return returns, advantages

    def train(self, states, actions, old_log_probs, rewards, dones):
        with torch.no_grad():
            old_action_probs, old_values = self.policy.forward(states)
            old_values = old_values.squeeze()

        returns, advantages = self.compute_advantages(rewards, old_values, dones)

        for _ in range(self.epochs):
            new_action_probs, new_values = self.policy.forward(states)
            #new_log_probs = torch.log(new_action_probs.gather(1, actions.unsqueeze(1)).squeeze())
            print(actions)
            new_log_probs = torch.log(torch.clamp(new_action_probs.gather(1, actions.unsqueeze(1)), min=1e-10)).squeeze()
            new_values = new_values.squeeze()

            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            clipped_ratios = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
            policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

            value_loss = self.mse_loss(new_values, returns)
            entropy_loss = -torch.mean(new_action_probs * torch.log(new_action_probs + 1e-10))

            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

            loss.backward(retain_graph=True)
            self.optimizer.step()

class Train:
    def __init__(self, ppo):
        self.ppo = ppo

    def run(self, X_train, y_train, epochs=100):
        env = IrisEnv(X_train, y_train)
        for episode in range(epochs):
            state = env.reset()
            log_probs, states, actions, rewards, dones = [], [], [], [], []

            for _ in range(len(y_train)):
                state_tensor = state.clone().detach().unsqueeze(0)
                action_probs, _ = self.ppo.policy(state_tensor)

                next_state, reward, done = env.step(action_probs)
                action = torch.multinomial(action_probs, 1).item()
                states.append(state_tensor.squeeze())
                actions.append(torch.tensor(action, device=device))
                rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
                dones.append(done)
                log_probs.append(torch.log(action_probs[0, action]))

                state = next_state
                if done:
                    break

            states = torch.stack(states)
            actions = torch.stack(actions)
            log_probs = torch.stack(log_probs)
            rewards = torch.stack(rewards)
            dones = torch.tensor(dones, dtype=torch.bool, device=device)

            print(states, actions)
            exit(0)
            self.ppo.train(states, actions, log_probs, rewards, dones)

            if episode % 10 == 0:
                print(f"Episode {episode} - PPO Training Step Completed")

    def predict(self, X_test):
        action_probs, state_value = self.ppo.policy(X_test)
        return action_probs

    def test(self, Y_pred, y_test):
        correct = 0
        for i in range(len(Y_pred)):
            action = torch.argmax(Y_pred[i]).item()
            if action == torch.argmax(y_test[i]).item():
                correct += 1
        accuracy = correct / len(y_test)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == '__main__':
    batch_size = 8

    iris = Iris()

    # Data
    X_train, X_test, Y_train, Y_test = iris.array_dataset(test_size=0.2, normilize=True)
    dataloader = iris.torch_dataset(batch_size, normilize=True)
    names = iris.class_names()

    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train = torch.tensor(Y_train, dtype=torch.long, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test = torch.tensor(Y_test, dtype=torch.long, device=device)

    # PPO train
    ppo = PPO(input_dim=4, output_dim=3)
    train = Train(ppo)
    y_train = train.run(X_train, Y_train, epochs=20)
    Y_pred = train.predict(X_test)
    train.test(Y_pred, Y_test)

    one_hot = torch.zeros_like(Y_pred)
    one_hot[torch.arange(Y_pred.shape[0]), torch.argmax(Y_pred, dim=1)] = 1
    one_hot = (one_hot == one_hot.max()).int()
    for a,b in zip(one_hot.int().tolist(), Y_test.tolist()):
        c = 'yes' if a == b else 'no'
        print(a, b, c)