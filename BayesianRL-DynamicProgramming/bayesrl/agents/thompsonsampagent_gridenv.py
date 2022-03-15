# this is the implementation of the thompson sampling based agent for the pomdp electric grid env

import numpy as np
from bayesrl.agents.modelbasedagent import ModelBasedAgent

class ThompsonSampAgentPOMDP(ModelBasedAgent):
    def __init__(self, observation_model, dirichlet_param, reward_param, **kwargs):
        super(ThompsonSampAgentPOMDP, self).__init__( **kwargs)
        self.dirichlet_param = dirichlet_param
        self.reward_param = reward_param
        self.reward = np.full((self.num_states, self.num_actions, self.num_states), self.reward_param)
        self.observation_model = observation_model
        self.reset_belief()
        self.__compute_policy()

    # initially the belief state is uniform distribution over all states
    def reset_belief(self):
        self.belief = np.array([1./self.num_states for _ in range(self.num_states)])

    def reset(self):
        super(ThompsonSampAgentPOMDP, self).reset()
        self.reset_belief()


    def interact(self, reward, observation, next_state_is_terminal, idx):
        # Handle start of episode.
        if reward is None:
            # Return random action since there is no information.
            next_action = np.random.randint(self.num_actions)
            self.last_action = next_action
            self.__observe(observation)
            return self.last_action

        # Handle completion of episode.
        if next_state_is_terminal:
            # Proceed as normal.
            pass

        for last_state,next_state in [(s,s_) for s in range(self.num_states) for s_ in range(self.num_states)]:
            tp = self.belief[last_state]*self.transition_probs[last_state,self.last_action,next_state]
            # Update the reward associated with (s,a,s') if first time.
            #if self.reward[last_state, self.last_action, next_state] == self.reward_param:
            self.reward[last_state, self.last_action, next_state] *= (1-tp)
            self.reward[last_state, self.last_action, next_state] += reward*tp

            # Update set of states reached by playing a.
            self.transition_observations[last_state, self.last_action, next_state] += tp

        # Update transition probabilities after every T steps
        if self.policy_step == self.T:
            self.__compute_policy()

        self.__update_belief(self.last_action,observation)
        # Choose next action according to policy.
        value_table = sum(self.belief[s]*self.value_table[s] for s in range(self.num_states))
        next_action = self._argmax_breaking_ties_randomly(value_table)

        self.policy_step += 1
        self.last_action = next_action

        return self.last_action

    def __compute_policy(self):
        """Compute an optimal T-step policy for the current state."""
        self.policy_step = 0
        self.transition_probs = np.zeros((self.num_states, self.num_actions, self.num_states))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.transition_probs[s,a] = np.random.dirichlet(self.transition_observations[s,a] +\
                                                            self.dirichlet_param, size=1)
        self._value_iteration(self.transition_probs)

    def __update_belief(self,action,observation):
        self.__transition(action)
        self.__observe(observation)


    # based on the transition update the belief (also called as action-based update belief)
    def __transition(self,action):
        for s in range(self.num_states):
            self.belief[s] = sum(self.transition_probs[s_,action,s]*self.belief[s_] for s_ in range(self.num_states))

    # this function update the value based on observation (also called as observation-based update belief)
    def __observe(self,observation):

        self.belief = [self.belief[s]*self.observation_model[s][observation] for s in range(self.num_states)]
        Z = sum(self.belief)
        self.belief = np.array(self.belief)/float(Z)

    def _value_iteration(self, transition_probs):
        """
        Run value iteration, using procedure described in Sutton and Barto
        (2012). The end result is an updated value_table, from which one can
        deduce the policy for state s by taking the argmax (breaking ties
        randomly).
        """
        value_dim = transition_probs.shape[0]
        value = np.zeros(value_dim)
        k = 0
        while True:
            diff = 0
            for s in range(value_dim):
                old = value[s]
                value[s] = np.max(np.sum(transition_probs[s]*(self.reward[s] +
                           self.discount_factor*np.array([value,]*self.num_actions)),
                           axis=1))
                diff = max(0, abs(old - value[s]))
            k += 1
            if diff < 1e-2:
                break
            if k > 1e6:
                raise Exception("Value iteration not converging. Stopped at 1e6 iterations.")
        for s in range(value_dim):
            self.value_table[s] = np.sum(transition_probs[s]*(self.reward[s] +
                   self.discount_factor*np.array([value,]*self.num_actions)),
                   axis=1)

    def _argmax_breaking_ties_randomly(self, x):
        """Taken from Ken."""
        max_value = np.max(x)
        indices_with_max_value = np.flatnonzero(x == max_value)
        return np.random.choice(indices_with_max_value)


    # here we will map the state float value to a decimal value
    def state_discretization(self,state):
        range = int((1.1 - 0.9) * 100)
        y = np.linspace(0.9 + 0.01, 1.1, range - 1)
        state_int = 0
        for ix, x in enumerate(state):
            # map the actual voltage to an integer
            # i.e. 0.9-0.91 val = 0, 0.91 - 0.92 val =1....etc
            int_val = np.digitize(x, y)
            state_int += int_val * (range ** ix)
        return state_int