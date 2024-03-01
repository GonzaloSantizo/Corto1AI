
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import json

def softmax(q_values, tau=1.0):
    """Compute softmax probabilities from q_values."""
    preferences = q_values / tau
    max_preference = np.max(preferences, axis=1, keepdims=True)
    exp_preferences = np.exp(preferences - max_preference)
    sum_exp_preferences = np.sum(exp_preferences, axis=1, keepdims=True)
    action_probs = exp_preferences / sum_exp_preferences
    return action_probs

def choose_action_softmax(state, q, tau=1.0):
 
    action_probs = softmax(q[state].reshape(1, -1), tau)
    action = np.random.choice(np.arange(q.shape[1]), p=action_probs.ravel())
    return action

def run(episodes, is_training=True, render=False):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open('frozen_lake8x8.json', 'r') as f:
            q = np.array(json.load(f))

    learning_rate_a = 0.9
    discount_factor_g = 0.9
    tau = 1.0  # Temperature parameter for softmax
    tau_decay = 0.99
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)
