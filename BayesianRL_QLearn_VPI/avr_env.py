# Create a Grid environment for the Bayesian RL agent and integrate the stochasticity of D(s,a)

from esa import SAW
import random
import pandas as pd

class GridEnv:
    def __init__(self, file_path, train_sets, action_space):
        self.file_path = file_path
        self.env = SAW(file_path)
        self.raw_train_sets = random.sample(train_sets, len(train_sets))
        self.train_sets = self.raw_train_sets
        self.action_space = action_space
        self.last_action = None

    def getState(self):
        output = None
        self.env.SolvePowerFlow()
        output = self.env.get_power_flow_results('bus').loc[:, 'BusPUVolt'].astype(float)
        return output.values[3:4]

    def execAction(self, action):
        data_list = []
        for i,value in enumerate([1,2,3]):
            data_list.append([value, '1', action[i]])
        output = self.env.ChangeParametersMultipleElement('gen', ['BusNum', 'GenID', 'GenVoltSet'], data_list)
        return output

    def nextSet(self):
        action = self.train_sets.pop(0)
        self.last_action = action
        self.env.ChangeParametersMultipleElement(action[0], action[1], action[2])
        try:
            output = self.getState()
        except Exception as e:
            pass
        return output

    def current_reward(self, state):
        violates = 0
        done = False
        for i in state:
            if i >= 1.05 or i <= 0.95:
                violates += 1
        done = True if violates == 0 else False
        reward = 50 - 100 * violates
        return reward, done

    def step(self, action):
        forcedone = False
        action_detail = self.action_space[action]
        self.execAction(action_detail)

        try:
            new_state = self.getState()
            reward, done = self.current_reward(new_state)
        except Exception as e:
            new_state = pd.DataFrame([-1] * 1)
            reward = -1000
            done = False
        if len(self.train_sets) == 0:
            forcedone = True
        return new_state, reward, done, forcedone

    def reset(self):
        action = self.last_action
        self.env.ChangeParametersMultipleElement(action[0], action[1], action[2])
        return self.nextSet()

    def resetGen(self):
        default_volts = [[1, '1', 1], [2, '1', 1], [3, '1', 1]]
        output = self.env.ChangeParametersMultipleElement('gen', ['BusNum', 'GenID', 'GenVoltSet'], default_volts)
        return output