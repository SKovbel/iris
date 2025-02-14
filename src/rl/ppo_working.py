import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        self.actor = nn.Linear(64, output_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

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
            _, old_values = self.policy(states)
            old_values = old_values.squeeze()

        returns, advantages = self.compute_advantages(rewards, old_values, dones)

        for _ in range(self.epochs):
            new_action_probs, new_values = self.policy(states)
            new_log_probs = torch.log(new_action_probs.gather(1, actions.unsqueeze(1)).squeeze())
            new_values = new_values.squeeze()

            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            clipped_ratios = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
            policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

            value_loss = self.mse_loss(new_values, returns)
            entropy_loss = -torch.mean(new_action_probs * torch.log(new_action_probs + 1e-10))

            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

            loss.backward(retain_graph=True)
            self.optimizer.step()

class IrisEnv:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.index = 0

    def reset(self):
        self.index = 0
        return self.X[self.index]

    def step(self, action):
        correct = (action == self.y[self.index]).item()
        reward = 1 if correct else -1
        self.index += 1
        done = self.index >= len(self.y)
        next_state = self.X[min(self.index, len(self.y) - 1)]  # Ensure valid index
        return next_state, reward, done


# Load and preprocess data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, dtype=torch.long, device=device)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.long, device=device)

# Initialize PPO
ppo = PPO(input_dim=4, output_dim=3)

# Train PPO
for episode in range(100):
    env = IrisEnv(X_train, y_train)
    state = env.reset()
    log_probs, states, actions, rewards, dones = [], [], [], [], []

    for _ in range(len(y_train)):
        state_tensor = state.clone().detach().unsqueeze(0)
        action_probs, _ = ppo.policy(state_tensor)
        action = torch.multinomial(action_probs, 1).item()

        next_state, reward, done = env.step(action)

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

    ppo.train(states, actions, log_probs, rewards, dones)

    if episode % 10 == 0:
        print(f"Episode {episode} - PPO Training Step Completed")

# Evaluate
correct = 0
for i in range(len(y_test)):
    state = X_test[i]
    state_tensor = state.unsqueeze(0)
    action_probs, _ = ppo.policy(state_tensor)
    action = torch.argmax(action_probs).item()

    if action == y_test[i].item():
        correct += 1

accuracy = correct / len(y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
