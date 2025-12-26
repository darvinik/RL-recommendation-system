import pickle
import matplotlib.pyplot as plt

with open("results.pkl", "rb") as f:
    data = pickle.load(f)

rewards = data["rewards"]
ctr = data["ctr"]

# ---- Plot CTR ----
plt.figure()
for name, values in ctr.items():
    plt.plot(values, label=name)

plt.xlabel("Episodes")
plt.ylabel("CTR")
plt.title("CTR over Time (Non-stationary Environment)")
plt.legend()
plt.savefig("ctr_comparison.png")
plt.show()

# ---- Plot Final Rewards ----
plt.figure()
plt.bar(rewards.keys(), rewards.values())
plt.ylabel("Average Total Reward")
plt.title("Average Reward Across Runs")
plt.savefig("reward_summary.png")
plt.show()


