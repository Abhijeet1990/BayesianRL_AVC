import itertools
import pickle
import random
from avr_env import GridEnv
from avr_rl_agent import RLAgentAVR
import numpy as np
import matplotlib.pyplot as plt

REWARD_HORIZON = 100
DISCOUNT_DECAY = 0.95


def createActionSpace():
    iterables = [[0.95,0.975,1.0,1.025, 1.05], [0.95,0.975,1.0,1.025, 1.05], [0.95,0.975,1.0,1.025, 1.05]]
    space = list(itertools.product(*iterables))
    return space


def calculate_expected_reward_TD(t, episodeSARs):
    expected_future_pnl = 0.
    discount_factor = DISCOUNT_DECAY

    for tau in range(1, REWARD_HORIZON):
        if t+tau < len(episodeSARs):
            expected_future_pnl += discount_factor * episodeSARs[t+tau][2]
            discount_factor *= DISCOUNT_DECAY

    if t+tau < len(episodeSARs):
        final_state = [episodeSARs[t+tau][0]]
        Q = agent.getMaxQ(final_state)
        return expected_future_pnl + DISCOUNT_DECAY * Q
    else:
        return expected_future_pnl

if __name__ == "__main__":
    train_sets = []
    for i in range(30000):
        data_set = [[5, '1', round(150*random.uniform(0.8,1.2),2)], [6, '1', round(185*random.uniform(0.8,1.2),2)], [8, '1', round(100*random.uniform(0.8,1.2),2)]]
        train_sets.append(['load', ['BusNum', 'LoadID', 'LoadMW'], data_set])

    file=open('wscc9_brl_train_set','wb+')
    pickle.dump(train_sets, file)
    file.close()
    action_space = createActionSpace()
    env = GridEnv(r"D:\Tutorials\RL_Tutorial\Case\WSCC_9_bus.pwb", train_sets, action_space)
    state_size = 1 # since we will be monitoring the voltage values of only the generator bus
    action_size = 125

    print("Training the agent...")
    # 2. train the agent on that trajectory, show that it learned some optimum
    agent = RLAgentAVR(state_size, action_size)

    # Training***************#
    training_episodes = 300
    train_agent = True
    training_pnls = []
    for j in range(1, 6):
        episodeSARs = []
        total_reward = 0
        for e in range(training_episodes):
            env.resetGen()
            #try:
            state = list(env.nextSet())
            done = False
            while not done:
                action = agent.act(state,train_agent)
                #print ('Action Taken ',action)
                next_state, reward, done, forcedone = env.step(action)
                total_reward += reward
                state = next_state
                if train_agent:
                    reward,done = env.current_reward(state)
                    sar = [state[0], action, reward]
                    episodeSARs.append(sar)
                    #print('storing episode ')
            for t in range(len(episodeSARs)):
                expected_future_pnl = calculate_expected_reward_TD(t, episodeSARs)
                reward_label = episodeSARs[t][2] + expected_future_pnl
                tmpSAR = [episodeSARs[t][0], episodeSARs[t][1], reward_label]
                agent.remember(tmpSAR)
            # except Exception as e:
            #     print(e)
            #     pass
            training_pnls.append(total_reward)
        agent.replay()

        pct_progress = (float(j) / float(5)) * 100.0
        if e == 0:
            print("pct_progress = %s %%" % (pct_progress))
        else:
            print("pct_progress = %s %% (current average P&L is %s)" % (pct_progress, np.mean(training_pnls)))





    # Testing*********************#
    train_agent = False
    testing_pnls = []
    testing_episodes = 300

    for j in range(1, 6):
        total_reward = 0
        for e in range(testing_episodes):
            env.resetGen()

            try:
                state = list(env.nextSet())
                done = False
                while not done:
                    action = agent.act(state, train_agent)
                    next_state, reward, done, forcedone = env.step(action)
                    total_reward += reward
                    state = next_state

                testing_pnls.append(total_reward)
            except:
                pass

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    ax1.plot(training_pnls)
    ax1.set_title("Training P&Ls")
    ax2.plot(testing_pnls)
    ax2.set_title("Validation P&Ls")
    plt.show()

    print ("Average out-sample P&L across the tests: ", np.mean(testing_pnls))