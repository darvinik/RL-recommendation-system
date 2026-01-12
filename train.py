from env.recommender_env import RecommenderEnv
from agents.epsilon_greedy import EpsilonGreedy
from agents.ucb import UCB
from agents.policy_gradient import PolicyGradientAgent
import numpy as np
import pickle

from env.recommender_env import RecommenderEnv
from agents.epsilon_greedy import EpsilonGreedy
from agents.ucb import UCB
from agents.policy_gradient import PolicyGradientAgent

# Environment

env = RecommenderEnv()
episodes = 5000
runs = 5

# Agents (DEFINE FIRST)
agents = {
    "Epsilon-Greedy": EpsilonGreedy(5),
    "UCB": UCB(5),
    "Policy Gradient": PolicyGradientAgent(5)
}

# Result containers
all_rewards = {k: [] for k in agents}
all_ctr = {k: [] for k in agents}

# Training
for name, agent in agents.items():
    run_rewards = []
    run_ctr = []

    for run in range(runs):
        total_reward = 0
        total_clicks = 0
        ctr_list = []

        for step in range(episodes):
            state, _ = env.reset()

            if name == "Policy Gradient":
                action, log_prob = agent.select_action(state)
                _, reward, _, _, _ = env.step(action)
                agent.update(log_prob, reward)
            else:
                action = agent.select_action()
                _, reward, _, _, _ = env.step(action)
                agent.update(action, reward)

            total_reward += reward
            total_clicks += reward
            ctr_list.append(total_clicks / (step + 1))

        run_rewards.append(total_reward)
        run_ctr.append(ctr_list)

    all_rewards[name] = np.mean(run_rewards)
    all_ctr[name] = np.mean(run_ctr, axis=0)

# Save results
with open("results.pkl", "wb") as f:
    pickle.dump({
        "rewards": all_rewards,
        "ctr": all_ctr
    }, f)

print("Training completed successfully.")
