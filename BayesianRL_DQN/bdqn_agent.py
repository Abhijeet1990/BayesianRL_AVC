import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from collections import namedtuple, deque
import torch.nn.functional as F
import torch.optim as optim
from model import QNetwork
import math



BUFFER_SIZE = 100000
BATCH_LENGTH = 64
learning_rate = 0.0005
#update_freq = 500
gamma = 0.99
tau = 0.001


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

replayMemory = deque([],maxlen=BUFFER_SIZE)

def sample(memory):
    """Randomly sample a batch of experiences from memory."""
    experiences = random.sample(memory, k=BATCH_LENGTH)
    states = torch.from_numpy(np.vstack([e[0]for e in experiences if e is not None])).float().to(device)
    actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(device)
    rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(device)
    next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(device)
    dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

    return (states, actions, rewards, next_states, dones)


class MCMCDQNAgent():
    def __init__(self, state_size, action_size, seed,samples=100000, update_freq=500):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.hidden = 16
        self.samples = samples
        self.t_step = 0
        self.update_freq = update_freq

        # Q-Network
        self.Q = QNetwork(state_size, action_size,self.hidden, seed).to(device)
        self.TargetQ = QNetwork(state_size, action_size, self.hidden,seed).to(device)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=learning_rate)

        # For keeping track of when to update
        self.counter = 0

    def SelectAction(self, state, epsilon =0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.Q.eval()
        with torch.no_grad():
            action_values = self.Q(state)
        self.Q.train()

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            return random.choice(np.arange(self.action_size))
        else:
            return np.argmax(action_values.cpu().data.numpy())

    # a function for computing the Root mean square error
    def rmse(self, predictions, targets):
        predictions = np.array(predictions)
        targets = np.array(targets)
        return np.sqrt(((predictions - targets) ** 2).mean())

    # updating the weights
    def set_weights(self,weights):
        s_size = self.state_size
        h_size = 16
        a_size = self.action_size
        fc1_end = (s_size * h_size) + h_size
        fc1_W = torch.from_numpy(weights[:s_size * h_size].reshape(s_size, h_size))
        fc1_b = torch.from_numpy(weights[s_size * h_size:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end + (h_size * a_size)].reshape(h_size, a_size))
        fc2_b = torch.from_numpy(weights[fc1_end + (h_size * a_size):])
        self.Q.fc1.weight.data.copy_(fc1_W.view_as(self.Q.fc1.weight.data))
        self.Q.fc1.bias.data.copy_(fc1_b.view_as(self.Q.fc1.bias.data))
        self.Q.fc2.weight.data.copy_(fc2_W.view_as(self.Q.fc2.weight.data))
        self.Q.fc2.bias.data.copy_(fc2_b.view_as(self.Q.fc2.bias.data))

    # loglikelihood computations
    def likelihood_func(self, fx, y, w, tausq):
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(np.array(y) - np.array(fx)) / tausq
        return [np.sum(loss), fx]

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq):
        h = self.hidden  # number hidden neurons
        d = self.state_size  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def TrainFromMemory(self, state, action, reward, next_state, done):

        # saving experiences to replay memory
        replayMemory.append((state, action, reward, next_state, done))

        # Learning at an update frequency
        self.counter = (self.counter + 1) % self.update_freq

        if self.counter == 0 and len(replayMemory) > BATCH_LENGTH:
            # sample transitions from replay memory
            transitions = sample(replayMemory)

            # extract the batch of transitions for Deep Q learning
            states, actions, rewards, next_states, dones = transitions

            # From the Q target network, obtain the Qmax for the next state
            Q_targets_next = self.TargetQ(next_states).detach().max(1)[0].unsqueeze(1)

            # Update the Q target for the current state for non-terminating transitions
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

            # Get expected Q values from original Q network
            Q_expected = self.Q(states).gather(1, actions)

            trainsize = len(transitions)
            y_train = Q_targets.numpy()

            w_size = (self.state_size * self.hidden) + (self.hidden * self.action_size) + self.hidden + self.action_size
            w = np.random.randn(w_size)
            step_w = 0.02
            step_eta = 0.01
            self.set_weights(w)

            Q_targets_next = self.TargetQ(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            y_train = Q_targets.numpy()

            Q_expected = self.Q(states).gather(1, actions)
            pred_train = Q_expected.detach().numpy()
            eta = np.log(np.var(np.array(pred_train) - np.array(y_train)))
            tau_pro = np.exp(eta)

            sigma_squared = 25
            nu_1 = 0
            nu_2 = 0
            prior_likelihood = self.prior_likelihood(sigma_squared, nu_1, nu_2, w,
                                                     tau_pro)  # takes care of the gradients
            [likelihood, pred_train] = self.likelihood_func(pred_train, y_train, w, tau_pro)

            n_accept = 0

            for i in range(self.samples - 1):
                w_proposed = w + np.random.normal(0, step_eta, w_size)
                eta_proposed = eta + np.random.normal(0, step_eta, 1)
                tau_proposed = math.exp(eta_proposed)

                self.set_weights(w_proposed)
                Q_targets_next = self.TargetQ(next_states).detach().max(1)[0].unsqueeze(1)
                Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
                y_train = Q_targets.numpy()
                Q_expected = self.Q(states).gather(1, actions)
                pred_train = Q_expected.detach().numpy()
                rmsetrain = self.rmse(pred_train, y_train)
                likelihood_proposal, pred_train = self.likelihood_func(pred_train, y_train, w_proposed, tau_proposed)

                # likelihood_ignore  refers to parameter that will not be used in the alg.
                prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposed, tau_proposed)
                diff_likelihood = likelihood_proposal - likelihood
                diff_priorliklihood = prior_prop - prior_likelihood
                mh_prob = min(0, (diff_likelihood + diff_priorliklihood))
                mh_prob = math.exp(mh_prob)
                u = random.uniform(0, 1)

                if u < mh_prob:
                    n_accept += 1
                    likelihood = likelihood_proposal
                    prior_likelihood = prior_prop
                    w = w_proposed
                    eta = eta_proposed
                    self.set_weights(w)
                    self.soft_update(self.Q, self.TargetQ, 1e-3)
                    if i % 100 == 0:
                        #print ( likelihood, prior_likelihood, rmsetrain, rmsetest, w, 'accepted')
                        print ('Sample:',i, 'RMSE train:', rmsetrain)

            print(n_accept, ' num accepted')
            print((n_accept * 100) / (self.samples * 1.0), '% was accepted')
            accept_ratio = n_accept / (self.samples * 1.0) * 100
            print("accept ratio=", accept_ratio)

    def TrainFromMemoryOrig(self, state, action, reward, next_state, done):

        # saving experiences to replay memory
        replayMemory.append((state, action, reward, next_state, done))

        # Learning at an update frequency
        self.counter = (self.counter + 1) % self.update_freq

        if self.counter == 0 and len(replayMemory) > BATCH_LENGTH:
            # sample transitions from replay memory
            transitions = sample(replayMemory)

            # extract the batch of transitions for Deep Q learning
            states, actions, rewards, next_states, dones = transitions

            # From the Q target network, obtain the Qmax for the next state
            Q_targets_next = self.TargetQ(next_states).detach().max(1)[0].unsqueeze(1)

            # Update the Q target for the current state for non-terminating transitions
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

            # Get expected Q values from original Q network
            Q_expected = self.Q(states).gather(1, actions)

            # Compute loss function based on the Q target and the estimate from the original Q
            loss = F.mse_loss(Q_expected, Q_targets)

            # Minimize the loss
            self.optimizer.zero_grad()

            # back propagate
            loss.backward()
            self.optimizer.step()

            # Update the target network
            tau = 0.001
            for TargetQ_param, Q_param in zip(self.TargetQ.parameters(), self.Q.parameters()):
                TargetQ_param.data.copy_(tau * Q_param.data + (1.0 - tau) * TargetQ_param.data)
