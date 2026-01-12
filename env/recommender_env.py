import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RecommenderEnv(gym.Env):
    def __init__(self, n_users=10, n_items=5):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items

        self.action_space = spaces.Discrete(n_items)
        self.observation_space = spaces.Discrete(n_users)

        # Click probabilities (user × item)
        self.click_prob = np.random.rand(n_users, n_items)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.user = np.random.randint(self.n_users)
        return self.user, {}

    def step(self, action):
        # Get click probability
        prob = self.click_prob[self.user, action]
        reward = 1 if np.random.rand() < prob else 0

        # Add preference drift (non-stationarity)
        drift = np.random.normal(0, 0.01, self.click_prob.shape)
        self.click_prob = np.clip(self.click_prob + drift, 0, 1)

        terminated = True
        truncated = False
        return self.user, reward, terminated, truncated, {}
