# This is the Bayesian RL agent for the automatic voltage regulation problem

import random
import numpy as np
from collections import deque
import pymc3 as pm
import matplotlib.pyplot as plt
#import theano.tensor as tt
import scipy.stats as st

# currently model the state space for may be only 3 gens. voltage value for simplicity
# discretize states into specific integer
# for instance assume
class RLAgentAVR:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.model = None
        self.MIN_VAL = 0.9
        self.MAX_VAL = 1.1
        self.D = int(int((self.MAX_VAL - self.MIN_VAL) * 100)) # lets just check for only 1 bus
        self.Qmus_estimates_mu = np.zeros((self.action_size, self.D)) * 200.
        self.Qmus_estimates_sd = np.ones((self.action_size, self.D)) * 200.
        self.Qsds_estimates_mu = np.ones((self.action_size, self.D)) * 10
        self.Qsds_estimates_sd = np.ones((self.action_size, self.D)) * 10


    def reset_memory(self):
        self.memory = deque(maxlen=20000)

    def remember(self, sar):
        self.memory.append(sar)

    def getMaxQ(self, state):

        if self.model is None:
            return 0.

        # In the Q-table we return the Q value of the best action (including flat, if both long and short are negative),
        # for the current state.
        action_slice = self.Qmus_estimates[:, self.state_value_to_index(state)]
        maxQ = np.max(action_slice, axis=0)

        return maxQ

    def act(self, state, use_explo=True):

        if self.model is None:
            return random.randrange(self.action_size)

        def calculate_VPI(mus, sds):

            def gain(i, i2, x):
                gains = []
                for j in range(len(mus)):
                    if j == i:
                        # special case: this is the best action
                        g = mus[i2] - np.minimum(x, mus[i2])
                    else:
                        g = np.maximum(x, mus[i]) - mus[i]

                    gains.append(g)

                gains = np.reshape(np.array(gains), [-1, len(x)]).transpose()
                return gains

            SAMPLE_SIZE = 1000
            Q_LOW = -1.
            Q_HIGH = 1.
            x = np.random.uniform(Q_LOW, Q_HIGH, SAMPLE_SIZE)
            x = np.reshape(x, [-1, 1])

            dist = st.norm(mus, np.exp(sds))

            probs = dist.pdf(x)

            best_action_idx = np.argmax(mus)

            tmp_mus = np.copy(mus)
            tmp_mus[best_action_idx] = -9999.
            second_best_action_idx = np.argmax(tmp_mus)

            gains = gain(best_action_idx, second_best_action_idx, x)

            return np.mean(gains * probs, axis=0)

        state_idx = self.state_value_to_index(state)
        state_mus = self.Qmus_estimates[:, state_idx]
        state_sds = self.Qsds_estimates[:, state_idx]

        if use_explo:
            VPI_per_action = calculate_VPI(state_mus, state_sds)
            action_scores = VPI_per_action + state_mus
            idx_selected_action = np.argmax(action_scores)

            return idx_selected_action
        else:
            return np.argmax(state_mus)

    # assume this to be a list of 2 p.u. voltages
    def state_value_to_index(self, s):
        range = int((self.MAX_VAL - self.MIN_VAL) * 100)
        y = np.linspace(self.MIN_VAL+0.01, self.MAX_VAL, range-1)
        state_int = 0
        for ix,x in enumerate(s):
            # map the actual voltage to an integer
            # i.e. 0.9-0.91 val = 0, 0.91 - 0.92 val =1....etc
            int_val = np.digitize(x,y)
            state_int += int_val*(range**ix)
        return state_int

    def replay(self):
        mem = np.array(self.memory)

        states = mem[:, :self.state_size]
        actions = np.reshape(mem[:, self.state_size], [-1, 1])
        rewards = mem[:, -1]

        full_tensor = []

        for t in range(len(states)):

            idx = self.state_value_to_index(states[t])

            full_tensor.append(np.array([actions[t], idx, rewards[t]]))

        # qvalues = [N x 3]
        # 1 - action index
        # 2 - state index
        # 3 - reward
        qvalues = np.array(full_tensor)

        with pm.Model() as self.model:

            Qmus = pm.Normal('Qmus', mu=self.Qmus_estimates_mu, sd=self.Qmus_estimates_sd, shape=[self.action_size, self.D])
            Qsds = pm.Normal('Qsds', mu=self.Qsds_estimates_mu, sd=self.Qsds_estimates_sd, shape=[self.action_size, self.D])

            idx0 = qvalues[:, 0].astype(int)
            idx1 = qvalues[:, 1].astype(int)

            #pm.Normal('likelihood', mu=Qmus[idx0, idx1], sd=np.exp(Qsds[idx0, idx1]), observed=qvalues[:, 2])
            pm.Normal('likelihood', mu=Qmus[idx0, idx1], sd=np.exp(Qsds[idx0, idx1]), observed=qvalues[:, 2])

            mean_field = pm.fit(n=15000, method='advi', obj_optimizer=pm.adam(learning_rate=0.1))
            # mean_field = pm.fit(n=15000, method='svgd',obj_optimizer=pm.adam(learning_rate=0.1))
            self.trace = mean_field.sample(5000)

        self.Qmus_estimates = np.mean(self.trace['Qmus'], axis=0)
        self.Qsds_estimates = np.mean(self.trace['Qsds'], axis=0)

        self.Qmus_estimates_mu = self.Qmus_estimates
        self.Qmus_estimates_sd = np.std(self.trace['Qmus'], axis=0)

        self.Qsds_estimates_mu = self.Qsds_estimates
        self.Qsds_estimates_sd = np.std(self.trace['Qsds'], axis=0)

        self.reset_memory()



