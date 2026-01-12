Reinforcement Learning Recommendation System

Overview:

This project implements a reinforcement learning–based recommendation system that learns from user feedback in an e-commerce-like setting.
The recommendation problem is formulated as a contextual bandit, where an agent recommends one item to a user and receives binary feedback (click or no-click).

The goal is to maximize cumulative reward and click-through rate (CTR) under changing user preferences.

Problem Formulation:

State (Context): User ID (simulated)

Action: Recommend one item from a fixed catalog

Reward:

1 if the user clicks

0 otherwise

Environment: Simulated user behavior with non-stationary preferences

Objective: Maximize long-term reward and CTR

Each interaction is a single-step episode, consistent with a contextual bandit setting.

Algorithms Implemented:

The following methods are implemented and compared:

Epsilon-Greedy:

Simple exploration strategy using random exploration

Upper Confidence Bound (UCB):

Uncertainty-aware exploration using confidence bounds

Policy Gradient:

Neural policy trained using online policy gradient updates (PyTorch)

These algorithms were chosen to study different exploration–exploitation trade-offs.

Key Enhancements:

To make the problem more realistic and evaluation meaningful:

Non-stationary environment:
User click probabilities drift over time to simulate changing preferences.

Multiple runs averaging:
Results are averaged across multiple runs to reduce randomness.

Business-relevant metrics:
Both CTR over time and total reward are reported.

Evaluation Metrics:

CTR (Click-Through Rate):
Measures recommendation quality over time.

Average Total Reward:
Measures cumulative value delivered across the full interaction horizon.

Both metrics are necessary to understand short-term quality and long-term impact.

Results Summary:

UCB achieves the highest average total reward and CTR in the non-stationary setting.

Epsilon-Greedy performs reasonably but wastes interactions due to random exploration.

Policy Gradient underperforms due to mismatch with a stateless, non-stationary bandit and sparse rewards.

These results align with theoretical expectations:
simpler bandit methods outperform deep RL when state and delayed rewards are absent.

Project Structure:
RL-recommendation-system/
├── agents/
│   ├── epsilon_greedy.py
│   ├── ucb.py
│   └── policy_gradient.py
├── env/
│   └── recommender_env.py
├── train.py
├── plots.py
├── requirements.txt
├── ctr_comparison.png
├── reward_summary.png
└── README.md

How to Run
1. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

2. Install dependencies
pip install -r requirements.txt

3. Train the agents
python train.py

4. Generate plots
python plots.py

Limitations

Users and items are simulated

No real user features are used

No delayed rewards or session modeling

No cold-start handling

These limitations are intentional to keep the focus on algorithm comparison and learning behavior.

Conclusion

This project demonstrates how reinforcement learning and bandit algorithms can be applied to recommendation systems and evaluated under non-stationary conditions.
The results highlight that algorithm-environment alignment is critical, and simpler methods can outperform deep RL in appropriate settings.

