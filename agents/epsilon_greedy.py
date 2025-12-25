import numpy as np

class EpsilonGreedy:
    def __init__(self, n_items, epsilon=0.1):
        self.epsilon = epsilon
        self.q = np.zeros(n_items)
        self.counts = np.zeros(n_items)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.q))
        return np.argmax(self.q)

    def update(self, action, reward):
        self.counts[action] += 1
        self.q[action] += (reward - self.q[action]) / self.counts[action]
