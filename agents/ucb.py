import numpy as np

class UCB:
    def __init__(self, n_items, c=2):
        self.c = c
        self.q = np.zeros(n_items)
        self.counts = np.zeros(n_items)
        self.t = 0

    def select_action(self):
        self.t += 1
        ucb_values = self.q + self.c * np.sqrt(
            np.log(self.t + 1) / (self.counts + 1e-5)
        )
        return np.argmax(ucb_values)

    def update(self, action, reward):
        self.counts[action] += 1
        self.q[action] += (reward - self.q[action]) / self.counts[action]
