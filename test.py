import math
import random
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
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


env = DialogueManagerEnv()

observation = env.reset()
n_observations = len(observation)
n_actions = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

BATCH_SIZE = 8
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def preprocess_item(item):
    emotions = torch.tensor(item['emotions'][0], dtype=torch.float32, device=device)
    sentences = item['sentences'][0].to(device)
    return torch.cat((emotions, sentences), dim=0)


def select_action(cur_state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    # EXPLOIT
    if sample > eps_threshold:
        with torch.no_grad():
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(preprocess_item(cur_state)).argmax().view(1, 1)
    # EXPLORE
    else:
        # pick a random action
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # This converts batch-array of Transitions to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states_list = [preprocess_item(item) for item in batch.next_state if item is not None]
    non_final_next_states = torch.stack(non_final_next_states_list)

    state_list = [preprocess_item(cur_state) for cur_state in batch.state]
    state_batch = torch.stack(state_list).to(device)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward, dim=0)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = ((next_state_values * GAMMA) + reward_batch).requires_grad_()

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    if any(param.grad is not None for param in policy_net.parameters()):
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


num_episodes = 2

episode_durations = []

# training loop
for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    # TODO state = wait for sentence from user
    # state = env.reset()
    state = env.state
    for t in count():
        action = select_action(state)
        # TODO wait for sentence from user
        new_sentence = 'Here is a sentence.'
        new_emotions = {'neutral': 0.9488840103149414, 'approval': 0.04927210882306099,
                        'realization': 0.01524870004504919,
                        'optimism': 0.007397462613880634, 'annoyance': 0.004410946741700172,
                        'confusion': 0.0038724469486624002, 'disapproval': 0.0034067267552018166,
                        'admiration': 0.002833909122273326, 'disappointment': 0.0022627972066402435,
                        'desire': 0.001386463176459074, 'curiosity': 0.001385977491736412,
                        'caring': 0.0013623429695144296,
                        'love': 0.0013163670664653182, 'disgust': 0.0012667339760810137,
                        'amusement': 0.0011411334853619337,
                        'sadness': 0.0010983888059854507, 'anger': 0.0009471763041801751,
                        'excitement': 0.0009354772046208382,
                        'fear': 0.0009152788552455604, 'joy': 0.0009133153362199664, 'gratitude': 0.0007278873817995191,
                        'surprise': 0.0005257704760879278, 'relief': 0.0003500702732708305,
                        'remorse': 0.0003163530782330781,
                        'embarrassment': 0.0003062635660171509, 'pride': 0.0002906423178501427,
                        'nervousness': 0.0002685450599528849, 'grief': 0.00024025565653573722}

        # RESHAPE THE INPUT INTO 1d VECTORS
        embeddings = []
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained("bert-base-uncased")
        encoded_input = tokenizer(new_sentence, return_tensors='pt')

        output = model(**encoded_input)
        new_sentence = output['last_hidden_state'].squeeze().reshape(-1)
        new_emotions = [value for value in new_emotions.values()]

        observation, done = env.step(action, new_sentence, new_emotions)
        # TODO wait for user input to calculate reward
        # reward will be +1 if right, and -1 if wrong
        # reward = torch.tensor([reward], device=device)
        reward = torch.tensor(1, device=device).view(1, 1)

        # Store the transition in memory
        memory.push(state, action, state, reward)

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        # target net weights = target net weights + policy net weights
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)
        # TODO takeout if you don't use episode_duration
        if done:
            episode_durations.append(t + 1)
            break
    print('Finished episode ' + str(i_episode))
