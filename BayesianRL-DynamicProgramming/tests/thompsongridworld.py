"""
Solves grid world using Thompson sampling.
"""

from bayesrl.environments import GridWorld
from bayesrl.agents.thompsonsampagent import ThompsonSampAgent
from bayesrl.trial import Trial
from bayesrl.plot import Plot

# Define environment.
gwenv = GridWorld(
    GridWorld.samples['larger'],
    action_error_prob=.1,
    rewards={'*': 50, 'moved': -1, 'hit-wall': -1})

num_trials = 1

# Define agent.
# Dirichlet params = 1, Reward params = 50

# compute the policy after every 'T' interval
ts_agent = ThompsonSampAgent(
    num_states=gwenv.num_states, num_actions=gwenv.num_actions,
    discount_factor=0.95, T=50, dirichlet_param=2, reward_param=50)

# pass the thompson agent and the grid world environment
trial_thompson1 = Trial(ts_agent, gwenv)

trial_thompson1.run_multiple(num_trials)

# Plots!
plot = Plot({"Thompson sampling": [trial_thompson1]})
plot.rewards_by_episode()
