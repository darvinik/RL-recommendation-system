import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNet(nn.Module):
    def __init__(self, n_items):
        super().__init__()
        self.fc = nn.Linear(1, n_items)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

class PolicyGradientAgent:
    def __init__(self, n_items):
        self.policy = PolicyNet(n_items)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)

    def select_action(self, state):
        state = torch.tensor([[state]], dtype=torch.float32)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, log_prob, reward):
        loss = -log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
