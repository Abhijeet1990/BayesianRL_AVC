
# Import Libraries
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
from BAC_Agent import learning_parameters
from GridEnv import GridEnv
import itertools
import random
from BAC_Agent import BAC_Agent

# Initialize the learning parameters
learning_params = learning_parameters()

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
ge = GridEnv(file_path=file_path,train_sets=train_sets,action_list=action_space,num_act=125,num_episode_eval = 5)

# Start BAC with learning parameters, BAC_domain and GYM environment
BAC_module = BAC_Agent(ge,learning_params)

perf, theta, pd_dataframe = BAC_module.BAC_Function()
# theta is the final learned policy

perf.to_csv(r'results\GridEnv_BAC_Evaluation.csv')
pd_dataframe.to_csv(r'results\GridEnv.csv')



# %%
# Visualize
# Apply the learned policy on the Gym render
random_state = ge.reset()
a, _ = ge.calc_score(theta, list(random_state))
done = False
episode_length = 300

# Render for num_update_max/sample_interval time i.e. 0-axis length of 'perf'
t = 0
while done == False or t < episode_length:  # in secs
    # We expect the agent to converge within minimum time steps with learned policy
    # 'theta'
    x_now, _, done, _ = ge.step(a)
    a, _ = ge.calc_score(theta, list(x_now))
    t += 1

input("Sim done. Press enter to close...")

# %% Data Plotting
plt.figure(0)
plt.plot(pd_dataframe[["Episode Batch"]], pd_dataframe[["Mean Squared Error"]], 'b-',
         label="MSE")
plt.plot(pd_dataframe[["Episode Batch"]], pd_dataframe[["Mean Absolute Error"]], 'r-',
         label="MAE")
plt.xlabel("BAC Batch")
plt.ylabel("MSE and MAE")
plt.legend()

plt.figure(1)
plt.plot(pd_dataframe[["Episode Batch"]], pd_dataframe[["Batch User Reward"]],
         'ro')
plt.xlabel("BAC Batch")
plt.ylabel("Avg. Batch Reward")

plt.figure(2)
plt.plot(pd_dataframe[["Episode Batch"]], pd_dataframe[["Avg. Episode Length (t)"]],
         'g*')
plt.xlabel("BAC Batch")
plt.ylabel("Avg. Episode Length (t)")
