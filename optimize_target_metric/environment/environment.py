import numpy as np
from sklearn.metrics import f1_score

import gym
from gym.spaces import Discrete, Box


class CustomEnv(gym.Env):
    def __init__(self, X, y, emb_size, n_classes):
        self.X, self.y = X, y

        self.indices = list(range(len(X)))
        np.random.shuffle(self.indices)

        self.reward_range = (-1.0, 1.0)
        self.action_space = Discrete(n_classes)  # {0, 1}
        self.observation_space = Box(
            low=-1.0, high=1.0, shape=(emb_size,), dtype=np.float32
        )  # [emb_size]

        self.pred = []
        self.real = []

    def reset(self):
        self.pred, self.real = [], []
        self.indices = list(range(len(self.X)))
        self.curr_idx = self.indices.pop()
        obs = self.X[self.curr_idx]
        return obs

    def step(self, action):
        self.pred.append(action)
        self.real.append(self.y[self.curr_idx])

        if not self.indices:
            reward = f1_score(self.real, self.pred, average="macro")
            reward = (reward - 0.5) * 2 * len(self.pred)
            return self.X[self.curr_idx], reward, 1, {}

        self.curr_idx = self.indices.pop()
        obs = self.X[self.curr_idx]
        return obs, 0, 0, {}

    def view(self):
        print(f1_score(self.real, self.pred, average="macro"))
