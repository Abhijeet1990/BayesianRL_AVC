# this is the implementation of the electric grid operation trials
import numpy as np

class Trial(object):
    """
    Class for running trial(s) for a given agent and task.

    Parameters
    ----------
    agent: Agent
    task: Task
    MIN_ITERATIONS: int
        The minimum number of iterations for a trial.
    MIN_EPISODES: int
        The minimum number of episodes for a trial.
    MAX_EPISODE_ITERATION: int
        The maximum number of iterations for each episode.
    """
    def __init__(self, agent, task, MIN_ITERATIONS=15000, MIN_EPISODES=8000, MAX_EPISODE_ITERATION=20):
        self.agent = agent
        self.task = task
        self.MIN_ITERATIONS = MIN_ITERATIONS
        self.MIN_EPISODES = MIN_EPISODES
        self.MAX_EPISODE_ITERATION = MAX_EPISODE_ITERATION

        self.array_rewards_by_episode = None
        self.array_rewards_by_iteration = None

    def run(self):
        iteration = episode = 0
        rewards_by_iteration = np.zeros(self.MIN_ITERATIONS)
        rewards_by_episode = np.zeros(self.MIN_EPISODES)
        self.agent.reset()

        while iteration < self.MIN_ITERATIONS or episode < self.MIN_EPISODES:
            print ("Episode:",episode)

            # This observe resets the system internally
            state = self.task.observe()
            reward = None
            cumulative_reward = 0
            episode_iteration = 0
            done = False
            while not done:
                # Tell the agent what happened and ask for a next action.
                action = self.agent.interact(reward, state, done, iteration)

                # Take action A, observe R, S'.
                state, reward, done, forced_done = self.task.step(action)

                # Log rewards.
                if iteration < self.MIN_ITERATIONS:
                    rewards_by_iteration[iteration] = reward
                cumulative_reward += reward

                iteration += 1
                episode_iteration += 1

            if episode < self.MIN_EPISODES:
                rewards_by_episode[episode] = cumulative_reward
            episode += 1
            print(' Episode Reward : ',cumulative_reward,', Episode Length : ',episode_iteration)

        return rewards_by_iteration, rewards_by_episode

    def run_multiple(self, num_trials):
        self.array_rewards_by_episode = np.zeros((num_trials, self.MIN_EPISODES))
        self.array_rewards_by_iteration = np.zeros((num_trials, self.MIN_ITERATIONS))
        for i in range(num_trials):
            self.array_rewards_by_iteration[i], self.array_rewards_by_episode[i] = self.run()

