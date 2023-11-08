import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DialogueManagerEnv(gym.Env):
    def __init__(self):
        super(DialogueManagerEnv, self).__init__()

        # 0: Ask for clarification, 1: Respond with an answer, 2: Give a list of answers
        self.action_space = spaces.Discrete(3)
        max_sentences = 200
        vector_size = 5404
        max_emotions = 28

        self.observation_space = spaces.Dict({
            'sentences': spaces.Box(low=-np.inf, high=np.inf, shape=(vector_size, max_sentences)),
            'emotions': spaces.Box(low=0, high=1, shape=(max_emotions,)),
        })

        self.state = {
            'sentences': [],
            'emotions': []
        }

        self.done = False

    def step(self, action, new_sentence=None, new_emotions=None):
        if self.done:
            raise Exception("Environment is already done. Call reset to start a new episode.")

        self.state['sentences'].append(new_sentence)
        self.state['emotions'].append(new_emotions)

        # Keep only the most recent 3 sentences
        self.state['sentences'] = self.state['sentences'][-3:]
        self.state['emotions'] = self.state['emotions'][-3:]

        return self.state, self.done

    def calculate_reward(self, action, correct_action):
        if action == correct_action:
            # If the actions match, provide a positive reward
            reward = 1.0
        else:
            # If the actions do not match, provide a negative reward
            reward = -1.0
        return reward

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        sentences = []
        emotions = []

        # Return the initial observation
        observation = {
            'sentences': sentences,
            'emotions': emotions,
        }

        return observation
