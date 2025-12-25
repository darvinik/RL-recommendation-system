from env.recommender_env import RecommenderEnv
from agents.epsilon_greedy import EpsilonGreedy
from agents.ucb import UCB
from agents.policy_gradient import PolicyGradientAgent
import pickle

env = RecommenderEnv()
episodes = 5000

agents = {
    "Epsilon-Greedy": EpsilonGreedy(5),
    "UCB": UCB(5),
    "Policy Gradient": PolicyGradientAgent(5)
}

rewards = {k: [] for k in agents}

for name, agent in agents.items():
    total_reward = 0
    for _ in range(episodes):
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
        rewards[name].append(total_reward)

with open("results.pkl", "wb") as f:
    pickle.dump(rewards, f)

print("Training completed. Results saved.")