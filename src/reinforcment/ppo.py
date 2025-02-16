import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_episods = 100
max_epochs = 100
max_series = 10
#
eta = 1e-5
epsilon=0.2
action_std=0.2
#
min_threshold = 0.35
max_threshold = 0.70
done_threshold = min_threshold
delta_threshold = (max_threshold - min_threshold) / max_episods

from dataset import Iris

class IrisEnv:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.index = 0
        self.sindex = 0
        self.indices = list(range(len(self.X)))

    def idx(self):
        return self.indices[self.index]

    def reset(self):
        self.index = 0
        self.sindex = 0
        self.indices = random.sample(self.indices, len(self.indices)) # shufle data each reset
        return self.X[self.idx()]

    def step(self, action):
        next_state, next_action = self.X[self.idx()], self.y[self.idx()]
        reward = self.reward(action, next_action)
        done = reward > done_threshold
        terminate = False

        self.sindex += 1
        if self.sindex >= max_series:
            self.index += 1
            self.sindex = 0
            terminate = True
        if self.index >= len(self.X):
            self.index = 0

        return next_state, reward, done, terminate

    def reward(self, action, Y):
        distance = torch.norm(action - Y, p=2)
        max_distance = torch.norm(torch.ones_like(action) - torch.zeros_like(action), p=2)
        reward = torch.clamp(1 - (distance / max_distance), 0, 1)
        return reward.item()

class Memory():
    def __init__(self):
        self.reset()

    def reset(self):
        self.actions = []
        self.states = []
        self.rewards = []
        self.dones = []
        self.logprobs = []
        self.terminats = []

    def push(self, state, action, reward, done, terminate, logprob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.terminats.append(terminate)
        self.logprobs.append(logprob.item())

    def pull(self):
        return torch.stack(self.states), \
               torch.stack(self.actions), \
               torch.tensor(self.rewards, dtype=torch.float32).unsqueeze(1), \
               torch.tensor(self.dones, dtype=torch.bool).unsqueeze(1), \
               torch.tensor(self.terminats, dtype=torch.bool).unsqueeze(1), \
               torch.tensor(self.logprobs, dtype=torch.float32).unsqueeze(1)

class ActorCritic(nn.Module):
    def __init__(self, state_size=4, action_size=3, hidden_size=32, action_std=0.2):
        super(ActorCritic, self).__init__()

        self.actor_fc1 = nn.Linear(state_size, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_fc3 = nn.Linear(hidden_size, hidden_size)
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.actor_sigma = nn.Linear(hidden_size, action_size)
        
        #self.critic_fc1 = nn.Linear(state_size, 2*hidden_size)
        #self.critic_fc2 = nn.Linear(2*hidden_size, 2*hidden_size)
        #self.critic_fc3 = nn.Linear(2*hidden_size, hidden_size)
        self.critic_fc1_b = nn.Linear(hidden_size, hidden_size)
        self.critic_value = nn.Linear(hidden_size, 1)

        self.action_var = torch.full((action_size,), action_std**2)

    def forward(self, state):
        x = torch.relu(self.actor_fc1(state))
        x = torch.relu(self.actor_fc2(x))
        x = torch.relu(self.actor_fc3(x))

        #v = torch.sigmoid(self.critic_fc1(state))
        #v = torch.sigmoid(self.critic_fc2(v))
        #v = torch.sigmoid(self.critic_fc3(v))
        v = torch.relu(self.critic_fc1_b(x))

        mean = torch.sigmoid(self.actor_mean(x))
        sigma = torch.sigmoid(self.actor_sigma(x)) + eta
        value = self.critic_value(v)

        return mean, sigma, value

    def predict(self, state):
        action_mean, action_sigma, state_value = self.forward(state)

        action_var = action_sigma ** 2
        #action_var = self.action_var.expand_as(action_mean)
        covariance_mat = torch.diag_embed(action_var)  
        distribution = MultivariateNormal(action_mean, covariance_mat)
        action = distribution.sample()
        action_logprob = distribution.log_prob(action)
        return action, action_logprob

    def evaluate(self, states, actions):
        action_mean, action_sigma, state_value = self.forward(states)
        #print('states', states.shape, 'actions', actions.shape, 'sigma', action_sigma.shape, 'xxx')

        action_var = action_sigma ** 2
        #action_var = self.action_var.expand_as(action_mean)
        covariance_mat = torch.diag_embed(action_var) 
        #print(action_var.shape, covariance_mat.shape, 'xxx')
        distribution = MultivariateNormal(action_mean, covariance_mat)
        action_logprob = distribution.log_prob(actions).unsqueeze(1)
        action_entropy = distribution.entropy().unsqueeze(1)
        #print(action_logprob.shape, action_entropy.shape, state_value.shape,  'yyy')
        return action_logprob, action_entropy, state_value

class PPO:
    def __init__(self, state_size=4, action_size=3, hidden_size=32, lr=0.001, gamma=0.99, epsilon=0.2, epochs=10, action_std=0.2):
        self.gamma = gamma
        self.epochs = epochs
        self.epsilon = epsilon

        self.policy = ActorCritic(state_size, action_size, hidden_size, action_std)
        self.policy_train = ActorCritic(state_size, action_size, hidden_size, action_std)

        self.policy_train.load_state_dict(self.policy.state_dict())

        self.mse_loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def discount(self, rewards, dones, terminates):
        returns  = []
        discount = 0
        for i in reversed(range(len(rewards))):
            if dones[i] or terminates[i]: 
                discount = 0
            discount = rewards[i] + self.gamma * discount
            returns.insert(0, discount)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        returns = (returns - returns.mean()) / (returns.std() + eta)
        return returns.unsqueeze(1)

    def predict(self, state):
        action, logprob = self.policy.predict(state)
        return action, logprob

    def train(self, states, actions, rewards, dones, terminates, logprobs):
        cost = 0.0
        returns = self.discount(rewards, dones, terminates)
        #advantages = returns - logprobs
        for _ in range(self.epochs):
            action_logprobs, action_entropy, state_values = self.policy_train.evaluate(states, actions)
            advantages = returns - state_values
            #print('states', states.shape, 'actions', actions.shape)
            #print('dones', dones.shape, 'term', terminates.shape, 'rewards', rewards.shape)
            #print(action_logprobs.shape, action_entropy.shape, state_values.shape)
            #print(returns.shape, state_values.shape, advantages.shape)
            ratios = torch.exp(action_logprobs - logprobs)
            clipped_ratios = torch.clamp(ratios, min=1-self.epsilon, max=1+self.epsilon)
            policy_loss = -torch.min(ratios*advantages, clipped_ratios*advantages)
            value_loss = self.mse_loss(state_values, returns).float()
            loss = policy_loss + 0.5*value_loss - 0.01*action_entropy
            self.optimizer.zero_grad()
            loss_mean = loss.mean()
            loss_mean.backward()
            self.optimizer.step()
            cost += loss_mean.item()
            #print(f"Policy Loss: {policy_loss.mean().item()}")
            #print(f"Value Loss: {value_loss.mean().item()}")
            #print(f"Entropy: {action_entropy.mean().item()}")
        self.policy.load_state_dict(self.policy_train.state_dict())

        return cost / self.epochs

class Train:
    def __init__(self, agent):
        self.agent = agent
        self.memory = Memory()

    def run(self, X_train, y_train, episods=100):
        global done_threshold, delta_threshold, max_episods

        env = IrisEnv(X_train, y_train)
        max_steps = max_series * len(y_train)
        for episode in range(episods):
            state = env.reset()
            for i in range(max_steps):
                state = state.clone()
                action, logprob = self.agent.predict(state)
                next_state, reward, done, terminate = env.step(action)
                self.memory.push(state, action, reward, done, terminate, logprob)
                state = next_state

            states, actions, rewards, dones, terminates, logprobs = self.memory.pull()
            #print(states.shape, actions.shape, rewards.shape, dones.shape, terminates.shape, logprobs.shape)
            self.memory.reset()


            start_time = time.time()
            cost = self.agent.train(states, actions, rewards, dones, terminates, logprobs)
            delta_time = time.time() - start_time


            if False:
                for i in range(len(dones)):
                    print(dones[i], terminates[i])
            if episode % (max_episods/10) == 0:
                reward_mean = rewards.mean().item()
                print(f"Episode: {episode}, time: {delta_time:.1f} seconds, cost: {cost:.7f}, reward mean: {reward_mean:.5f}, threshold: {done_threshold:.2f}")
            done_threshold = min(done_threshold + delta_threshold, 1)

    def predict(self, X_test):
        action_probs, _, _ = self.agent.policy(X_test)
        return action_probs

    def test(self, Y_pred, y_test):
        correct = 0
        for i in range(len(Y_pred)):
            action = torch.argmax(Y_pred[i]).item()
            if action == torch.argmax(y_test[i]).item():
                correct += 1
        accuracy = correct / len(y_test)
        print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == '__main__':
    iris = Iris()

    # Data
    X_train, X_test, Y_train, Y_test = iris.array_dataset(test_size=0.2, normilize=True)
    names = iris.class_names()

    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train = torch.tensor(Y_train, dtype=torch.long, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test = torch.tensor(Y_test, dtype=torch.long, device=device)

    # PPO train
    ppo = PPO(epsilon=epsilon, epochs=max_epochs, action_std=action_std)
    train = Train(ppo)
    y_train = train.run(X_train, Y_train, episods=max_episods)
    Y_pred = train.predict(X_test)
    train.test(Y_pred, Y_test)
    print(Y_pred)
    one_hot = torch.zeros_like(Y_pred)
    one_hot[torch.arange(Y_pred.shape[0]), torch.argmax(Y_pred, dim=1)] = 1
    one_hot = (one_hot == one_hot.max()).int()
    for a,b in zip(one_hot.int().tolist(), Y_test.tolist()):
        c = 'yes' if a == b else 'no'
        print(a, b, c)