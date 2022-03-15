# this would be the final implemnetation of training the thompson sampling agent
from bayesrl.environments import pomdpgridenv
from bayesrl.agents.thompsonsampagent_gridenv import ThompsonSampAgentPOMDP
from bayesrl.trial_gridenv import Trial
from bayesrl.plot import Plot
import random
import itertools

def createActionSpace():
    iterables = [[0.95,0.975,1.0,1.025, 1.05], [0.95,0.975,1.0,1.025, 1.05], [0.95,0.975,1.0,1.025, 1.05]]
    space = list(itertools.product(*iterables))
    return space

train_sets = []
for i in range(30000):
    data_set = [[5, '1', round(150*random.uniform(0.8,1.2),2)], [6, '1', round(185*random.uniform(0.8,1.2),2)], [8, '1', round(100*random.uniform(0.8,1.2),2)]]
    train_sets.append(['load', ['BusNum', 'LoadID', 'LoadMW'], data_set])


file_path = r"D:\Tutorials\RL_Tutorial\Case\WSCC_9_bus.pwb"
action_space = createActionSpace()

# Define environment.
task = pomdpgridenv.GridEnvPOMDP(file_path,train_sets,action_space)

num_trials = 1

################################################################################
# Thompson Sampling
################################################################################
# Dirichlet params = 1, Reward params = 50
agent = ThompsonSampAgentPOMDP(observation_model=task.getDistribution(),
    num_states=task.num_states, num_actions=task.num_actions,
    discount_factor=0.95, T=50, dirichlet_param=1, reward_param=50)
trial_thompson1 = Trial(agent, task, MIN_EPISODES=30)
trial_thompson1.run_multiple(num_trials)

plot = Plot({"Thompson sampling": [trial_thompson1]#, trial_thompson2, trial_thompson3]
            })
# Plot cumulative rewards by iteration
plot.cum_rewards_by_iteration()
# Plot rewards by episode
plot.rewards_by_episode()
