import math
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from DialogueManagerEnv import DialogueManagerEnv

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(5404, 128)
        self.layer2 = nn.Linear(128, 16)
        self.layer3 = nn.Linear(16, num_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class RLModel:

    BATCH_SIZE = 8
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        self.env = DialogueManagerEnv()

        self.env.reset()
        n_actions = self.env.action_space.n

        self.policy_net = DQN(n_actions).to(RLModel.device)
        self.target_net = DQN(n_actions).to(RLModel.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=RLModel.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.last_action = None
        self.last_sentence = None
        self.last_emotions = None

    def select_action(self, cur_state):
        sample = random.random()
        eps_threshold = RLModel.EPS_END + (RLModel.EPS_START - RLModel.EPS_END) * \
                        math.exp(-1. * self.steps_done / RLModel.EPS_DECAY)
        self.steps_done += 1
        # EXPLOIT
        if sample > eps_threshold:
            with torch.no_grad():
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(cur_state).argmax().view(1, 1)
        # EXPLORE
        else:
            # pick a random action
            return torch.tensor([[self.env.action_space.sample()]], device=RLModel.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < RLModel.BATCH_SIZE:
            return
        transitions = self.memory.sample(RLModel.BATCH_SIZE)
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=RLModel.device, dtype=torch.bool)
        non_final_next_states_list = [item for item in batch.next_state if item is not None]
        non_final_next_states = torch.stack(non_final_next_states_list)

        state_list = [cur_state for cur_state in batch.state]
        state_batch = torch.stack(state_list).to(RLModel.device)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward, dim=0)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(RLModel.BATCH_SIZE, device=RLModel.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = ((next_state_values * RLModel.GAMMA) + reward_batch).requires_grad_()

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        if any(param.grad is not None for param in self.policy_net.parameters()):
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    # can't call again before calling process_reward
    def process_message(self, new_sentence, new_emotions):
        self.last_sentence = new_sentence
        self.last_emotions = new_emotions
        self.last_action = self.select_action(self.env.state)

        return self.last_action[0, 0].item()

    # has to come directly after process message
    def process_reward(self, correct_action):
        # reward will be +1 if right, and -1 if wrong
        if correct_action == self.last_action:
            reward = torch.tensor(1, device=RLModel.device).view(1, 1)
        else:
            reward = torch.tensor(-1, device=RLModel.device).view(1, 1)

        self.env.step(self.last_action, self.last_sentence, self.last_emotions)

        # Store the transition in memory
        # the next state will be the same as the current state
        self.memory.push(self.env.state, self.last_action, self.env.state, reward)

        # Perform one step of the optimization (on the policy network)
        self.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        # target net weights = target net weights + policy net weights
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = (policy_net_state_dict[key] * RLModel.TAU +
                                          target_net_state_dict[key] * (1 - RLModel.TAU))
        self.target_net.load_state_dict(target_net_state_dict)
