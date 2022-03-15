import gym
from gym import spaces
from gym.utils import seeding
from esa import SAW
import numpy as np
import random
from scipy.spatial import distance
import pandas as pd

class GridEnv(gym.Env):
    def __init__(self, file_path, train_sets,action_list,num_act,levels =20,num_episode_eval = 10):
        self.file_path = file_path
        self.env = SAW(file_path)
        self.raw_train_sets = random.sample(train_sets, len(train_sets))
        self.train_sets = self.raw_train_sets
        self.NUM_ACT = num_act
        self.NUM_STATE_FEATURES = levels
        self.action_space = spaces.Discrete(self.NUM_ACT)
        self.last_action = None
        self.levels = levels
        self.min_volt = 0.9
        self.max_volt = 1.1
        self.phi_x = np.zeros((self.NUM_STATE_FEATURES, 1))
        # since we will be observing two nodes in the problem the array size of low and high are 1
        self.low = np.array([self.min_volt], dtype=np.float32)
        self.high = np.array([self.max_volt], dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.VOLT_CENTERS = np.zeros(self.NUM_STATE_FEATURES, dtype=np.float32)
        for i in range(0, self.NUM_STATE_FEATURES):
            self.VOLT_CENTERS[i] = 0.905 + i*0.01
        self.sig_grid = 1.3 * float((self.max_volt - self.min_volt)/ self.levels)
        self.sig_grid2 = self.sig_grid ** 2
        self.SIG_GRID = self.sig_grid2 * np.identity(1)
        self.INV_SIG_GRID = np.linalg.inv(self.SIG_GRID)
        self.num_policy_param = self.NUM_STATE_FEATURES * self.NUM_ACT
        self.STEP = 1
        self.prng = np.random.RandomState()
        self.prng.seed(2)
        self.ACT = np.arange(0,self.NUM_ACT)
        self.num_episode_eval = num_episode_eval
        self.action_list = action_list

    def getState(self):
        output = None
        self.env.SolvePowerFlow()
        output = self.env.get_power_flow_results('bus').loc[:, 'BusPUVolt'].astype(float)

        # instead of returning the exact state it need to return a distribution of state
        return_state = output.values[3:4]
        all_states = output.values
        return return_state, all_states

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
        action_detail = self.action_list[action]
        self.execAction(action_detail)

        try:
            new_state,all_state = self.getState()
            reward, done = self.current_reward(new_state,all_state)
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
        return self.nextSet()

    def calc_score(self, theta, state):
        """
        Description-----
        calc_score: Chooses the next action 'a' and computes the Fisher Information
        matrix score 'scr' for the grid_env.
        Parameters------
        theta: Current policy
        state: Current State = [x]
        Return-----
        a: Action to move to next state
        scr: Fisher Information matrix
        """

        y = state[0]

        # feature values
        phi_x = np.zeros((self.NUM_STATE_FEATURES, 1))
        mu = np.zeros(self.NUM_ACT)
        tmp1 = np.zeros((1, 1))

        for tt in range(0, self.NUM_STATE_FEATURES):
            tmp1 = y - self.VOLT_CENTERS[tt].reshape((1, 1))
            # Turns out to be a scalar
            # We solve x'Ax by Matmul Ax then dot product of x and Ax
            arbi1 = np.dot(self.INV_SIG_GRID, tmp1)
            phi_x[tt, 0] = np.exp(-0.5 * np.dot(np.transpose(tmp1), arbi1)).item()

        for i in range(0, self.NUM_ACT):
            for j in range(0,self.NUM_ACT):
                if i == j:
                    if i == 0:
                        phi_xa = np.vstack(phi_x)
                    else:
                        phi_xa = np.vstack((phi_xa, phi_x))
                else:
                    if i != 0:
                        if phi_xa is None:
                            phi_xa = np.vstack(np.zeros((self.NUM_STATE_FEATURES, 1)))
                        else:
                            phi_xa = np.vstack((phi_xa, np.zeros((self.NUM_STATE_FEATURES, 1))))
                    else:
                        phi_xa = np.vstack((phi_xa,np.zeros((self.NUM_STATE_FEATURES, 1))))

            # lol = np.dot(np.transpose(phi_xa), theta)
            lol = np.dot(np.transpose(phi_xa), np.vstack(theta))

            phi_xa = None

            mu[i] = np.exp(lol.item())

        mu = mu / sum(mu)

        tmp2 = self.prng.rand()

        # Added some randomness is the action value. a * tmp2
        for i in range(0, self.NUM_ACT):

            if (tmp2 < (i+1)*(1/self.NUM_ACT) and tmp2 > i*(1/self.NUM_ACT)):
                a = self.ACT[i]
                for j in range(0, self.NUM_ACT):
                    if i == j:
                        if i == 0:
                            scr = np.vstack(-phi_x * mu[i])
                        else:
                            scr = np.vstack((scr, -phi_x * mu[i]))
                    else:
                        if i != 0:
                            if scr is None:
                                scr = np.vstack(phi_x * (1 - mu[j]))
                            else:
                                scr = np.vstack((scr, phi_x * (1 - mu[j])))
                        else:
                            scr = np.vstack((scr,phi_x * (1 - mu[j])))
                break
            else:
                scr = None
                continue

        # scr HAS TO BE a row-wise 2D array of size num_state_features x 1
        return a, scr


    def kernel_kx(self, state, statedic):
        sigk_x = 1
        ck_x = 1
        x = state[0]
        xdic = []
        # Possible conflict at concatenation
        for i in range(0, len(statedic)):
            xdic.append(statedic[i][0].reshape((1, 1))) ## The shape is v-important
        xdic = np.hstack(xdic)
        arbitrary = np.vstack([1.0])
        y = np.multiply(arbitrary, x)### We will see
        ydic = np.multiply(np.tile(arbitrary, (1, np.shape(xdic)[1])),  xdic)
        # Element-wise squaring of Euclidean pair-wise distance
        #Need to install pdist python package
        temp = np.square(distance.cdist(np.transpose(y), np.transpose(ydic)))
        kx = ck_x * np.exp((-1 * temp) / (2 * sigk_x*sigk_x))
        return np.squeeze(kx)


    def kernel_kxx(self):
        kxx = 1
        return kxx


    def perf_eval(self, theta, learning_params):
        """
        Evaluates the policy after every n(sample_interval) (e.g. 50) updates.
        See BAC.py for the function call protocol --> Find --> perf_eval
        """
        step_avg = 0

        for l in range(0, self.num_episode_eval):
            t = 0
            env_current_state = self.resetGen()
            state = env_current_state
            # Since Gym.reset() only returns state = (position, velocity)
            # and we also need a C map for this state which is necessary for
            # BAC computation and is exclusive for each environment and observations

            done = False
            a, _ = self.calc_score(theta, state)
            reward2 = 0
            reward1 = 0
            while done == 0 or t < learning_params.episode_len_max:
                for istep in range(self.STEP):
                    if done == 0:
                        # state, _ = self.dynamics(state, a, self)
                        # state = self.is_goal(state, self)
                        state, reward, done, _ = self.step(a)  ### Fix this array methods
                        state = list(state)
                        state.append(done)
                        reward1 += reward  ## Reward accumulated by Gym
                        reward2 -= 1  ## User defined reward
                a, _ = self.calc_score(theta, state)
                t = t + 1

            step_avg = step_avg + t

        perf = step_avg / self.num_episode_eval

        return perf, reward1, reward2

    def calc_reward(self, state):
        violates = 0
        if state >= 1.05 or state <= 0.95:
            violates += 1
        reward = 50 - 100 * violates
        return reward
