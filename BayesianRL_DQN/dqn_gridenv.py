import numpy as np
import pandas as pd
from esa import SAW
import random
import pickle
import itertools
from collections import deque
from bdqn_agent import MCMCDQNAgent
from dqn_agent import Agent
import torch
import matplotlib.pyplot as plt

class GridEnv:
    def __init__(self, file_path, train_sets, action_space):
        self.file_path = file_path
        self.env = SAW(file_path)
        self.raw_train_sets = random.sample(train_sets, len(train_sets))
        self.train_sets = self.raw_train_sets
        self.action_space = action_space
        self.last_action = None

    def getState(self):
        self.env.SolvePowerFlow()
        output = self.env.get_power_flow_results('bus').loc[:, 'BusPUVolt'].astype(float)
        return output

    def execAction(self, action):
        data_list = []
        for i, value in enumerate([1, 2, 3]):
            data_list.append([value, '1', action[i]])
        output = self.env.ChangeParametersMultipleElement('gen', ['BusNum', 'GenID', 'GenVoltSet'], data_list)
        return output

    def nextSet(self):
        action = self.train_sets.pop(0)
        self.last_action = action
        self.env.ChangeParametersMultipleElement(action[0], action[1], action[2])
        physical_states = self.getState()
        return physical_states

    def isInViolation(self, state, cyber_state):
        state_nparray = state.values
        violate = True if np.min(state_nparray) <= 0.95 or np.max(state_nparray) >= 1.05 else False
        return violate

    def getReward(self, state):
        state_nparray = state.values
        violates = 0
        done = False
        for i in state_nparray:
            if i >= 1.05 or i <= 0.95:
                violates += 1
        done = True if violates == 0 else False
        reward = 199 - 100 * violates
        return reward, done

    def step(self, action):
        forcedone = False
        action_detail = self.action_space[action]
        self.execAction(action_detail)
        #new_physical_state = self.getState()
        #         reward, done = self.getReward(new_physical_state, self.last_cyber_state, action_detail)
        try:
            new_physical_state = self.getState()
            reward, done = self.getReward(new_physical_state)

        except Exception:
            new_physical_state = pd.DataFrame([-1] * 9)
            reward = -1000
            done = False

        if len(self.train_sets) == 0:
            forcedone = True

        return new_physical_state, reward, done, forcedone

    def reset(self):
        action = self.last_action
        output = self.env.ChangeParametersMultipleElement(action[0], action[1], action[2])
        return self.nextSet()

    def resetGen(self):
        default_volts = [[1, '1', 1], [2, '1', 1], [3, '1', 1]]
        output = self.env.ChangeParametersMultipleElement('gen', ['BusNum', 'GenID', 'GenVoltSet'], default_volts)
        return output


def createActionSpace():
    iterables = [[0.95,0.975,1.0,1.025, 1.05], [0.95,0.975,1.0,1.025, 1.05], [0.95,0.975,1.0,1.025, 1.05]]
    space = list(itertools.product(*iterables))
    return space

def dqn(n_episodes=1000, max_t=500, eps_start=1.0, eps_end=0.01, eps_decay=0.995, BDQN = False, update_freq = 500):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    train_sets = []
    for i in range(30000):
        data_set = [[5, '1', round(150 * random.uniform(0.7, 1.3), 2)],
                    [6, '1', round(185 * random.uniform(0.7, 1.3), 2)],
                    [8, '1', round(100 * random.uniform(0.7, 1.3), 2)]]
        train_sets.append(['load', ['BusNum', 'LoadID', 'LoadMW'], data_set])


    file=open('wscc9_train_set','wb+')
    pickle.dump(train_sets, file)
    file.close()
    action_space = createActionSpace()
    env = GridEnv(r"D:\Tutorials\RL_Tutorial\Case\WSCC_9_bus.pwb", train_sets, action_space)
    state_size = 9
    action_size = 125

    #agent = MCMCDQNAgent(state_size=state_size, action_size=action_size, seed=0,update_freq=update_freq)
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = env.resetGen()
        score = 0
        state = env.nextSet()
        state = state.to_numpy()
        for t in range(max_t):
            #print(f"{t} time step of episode {i_episode}")
            #action = agent.SelectAction(state, eps)
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.to_numpy()
            # if BDQN:
            #     agent.TrainFromMemory(state,action,reward,next_state,done)
            # else:
            #     agent.TrainFromMemoryOrig(state,action,reward,next_state,done)
            agent.step(state,action,reward,next_state,done)
            state = next_state
            score += reward
            if done:
                break
        print("score=", score)
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            if BDQN:
                torch.save(agent.Q.state_dict(), 'bdqn_checkpoint_500_1L.pth')
            else:
                torch.save(agent.Q.state_dict(), 'dqn_checkpoint_100.pth')
            break
    return scores


scores = dqn(BDQN=False,update_freq=500)

# Smoothing over a window
def avg_window_smooth(values, window_size = 10):
    smoothed = []
    i=0
    while i < len(values) - window_size + 1:
        curr_window = values[i: i+ window_size]
        avg = sum(curr_window)/ window_size
        smoothed.append(avg)
        i+=1
    return smoothed



#plt.plot(avg_window_smooth(scores,50))

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
#plt.plot(np.arange(len(scores)), scores)
plt.plot(np.arange(len(avg_window_smooth(scores,50))),avg_window_smooth(scores,50))
plt.ylabel('Score')
plt.xlabel('Episode #')
fig.savefig('avg_reward_BDQN_update_freq_500steps_1L.jpg',  dpi=150)
plt.show()


