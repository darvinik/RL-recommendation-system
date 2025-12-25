import pickle
import matplotlib.pyplot as plt

with open("results.pkl", "rb") as f:
    rewards = pickle.load(f)

for name, values in rewards.items():
    plt.plot(values, label=name)

plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.title("RL Recommendation Performance Comparison")
plt.legend()
plt.savefig("comparison.png")
plt.show()

