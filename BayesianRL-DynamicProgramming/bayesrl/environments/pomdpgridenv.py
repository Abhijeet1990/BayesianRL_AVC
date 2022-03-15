# This is the POMDP implementation of the electric grid

from esa import SAW
import random
import pandas as pd
import numpy as np
from ..utils import check_random_state

class GridEnvPOMDP:
    def __init__(self, file_path, train_sets, action_space, true_obs_prob = 0.8,levels=20,action_error_prob = 0.1,random_state=None):
        self.file_path = file_path
        self.env = SAW(file_path)
        self.raw_train_sets = random.sample(train_sets, len(train_sets))
        self.train_sets = self.raw_train_sets
        self.action_space = action_space
        self.last_action = None
        self.true_obs_prob = true_obs_prob
        self.levels = levels
        self.num_actions = len(self.action_space)
        self.num_states = levels # the state can be any one of the discrete level
        self.obs_model = self.getDistribution()
        self.action_error_prob = action_error_prob
        self.random_state = check_random_state(random_state)

    def getState(self):
        output = None
        self.env.SolvePowerFlow()
        output = self.env.get_power_flow_results('bus').loc[:, 'BusPUVolt'].astype(float)

        # instead of returning the exact state it need to return a distribution of state
        return_state = output.values[3:4]
        all_states = output.values
        return return_state, all_states

    def getDistribution(self):
        # should return the observation model used by the thompson agent
        # The state can take any value between say [0-19].. so the idea is like
        # 0:.9-.91 1: .91-.92 ............ 19 : 1.09 - 1.1
        # say the agent doesnt observe the exact value...
        # say for the states in between 0-5 and 15-19 where the voltages are beyond limit, the attacker cause less variance
        # while in between 5-12 it causes high variance
        # if say the state is 10, that means the agent will see this state with true_pos, 9 and 11 with .03 each say and rest other (.14)
        # if say the state is 4, that means the agent will see the state with true_pos (.8), 3 and 5 with 0.07 each and rest (.06)
        observation_model = []
        for i in range(self.levels):
            per_level_model =list(np.zeros(self.levels))
            for j in range(self.levels):
                if i < 5 or i > 14:  # cause less variance
                    if j == i:
                        per_level_model[j] = self.true_obs_prob
                    elif (j == (i+1) or j == (i-1)):
                        per_level_model[j] = float((1.0 - self.true_obs_prob - .06)/2.0)
                    else:
                        per_level_model[j] = float((.06)/float(self.levels - 3))
                else:   # cause high variance
                    if j == i:
                        per_level_model[j] = self.true_obs_prob
                    elif (j == (i+1) or j == (i-1)):
                        per_level_model[j] = float((1.0 - self.true_obs_prob - .14)/2.0)
                    else:
                        per_level_model[j] = float((.14)/float(self.levels - 3))
            observation_model.append(per_level_model)
        return observation_model

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
            output,_ = self.getState()
        except Exception as e:
            pass
        return output

    def current_reward(self, state,all_state):
        violates = 0
        done = False
        for i in all_state:
            if i >= 1.05 or i <= 0.95:
                violates += 1
        done = True if violates == 0 else False
        reward = 50 - 100 * violates
        return reward, done

    def step(self, action):
        forcedone = False
        if self.action_error_prob and self.random_state.rand() < self.action_error_prob:
            action = self.random_state.choice(self.num_actions)
        action_detail = self.action_space[action]
        self.execAction(action_detail)

        try:
            new_state,all_state = self.getState()
            reward, done = self.current_reward(new_state,all_state)
            new_st = self.state_discretization(new_state)
            obs_dist = self.obs_model[new_st]
            obs_mult_dist = np.random.multinomial(1, obs_dist)
            new_state = obs_mult_dist.tolist().index(1)
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

    def observe(self):
        default_volts = [[1, '1', 1], [2, '1', 1], [3, '1', 1]]
        self.env.ChangeParametersMultipleElement('gen', ['BusNum', 'GenID', 'GenVoltSet'], default_volts)
        action = self.train_sets.pop(0)
        self.last_action = action
        self.env.ChangeParametersMultipleElement(action[0], action[1], action[2])
        try:
            output, _ = self.getState()
        except Exception as e:
            pass
        st = self.state_discretization(output)
        obs_dist = self.obs_model[st]
        obs = np.random.multinomial(1, obs_dist)
        return obs.tolist().index(1)


    # here we will map the state float value to a decimal value
    def state_discretization(self, state):
        range = int((1.1 - 0.9) * 100)
        y = np.linspace(0.9 + 0.01, 1.1, range - 1)
        state_int = 0
        for ix, x in enumerate(state):
            # map the actual voltage to an integer
            # i.e. 0.9-0.91 val = 0, 0.91 - 0.92 val =1....etc
            int_val = np.digitize(x, y)
            state_int += int_val * (range ** ix)
        return state_int

