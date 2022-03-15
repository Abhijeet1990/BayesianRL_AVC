import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim
import math

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 500  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 500  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, samples=50000):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.hidden_size = 16
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, self.hidden_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, self.hidden_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.samples = samples

        self.topology = [self.state_size, self.hidden_size, self.action_size]

    # =============================================================================
    #         print(self.qnetwork_local.state_dict())
    #         raise Exception("sdfadf")
    # =============================================================================

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_li fc2_W = torch.from_numpy(weights[fc1_end:fc1_end+(h_size*a_size)].reshape(h_size, a_size))
        fc2_b ke): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def set_weights(self, weights):
        s_size = self.state_size
        h_size = 16
        a_size = self.action_size
        #        print(weights.shape)
        # separate the weights for each layer
        fc1_end = (s_size * h_size) + h_size
        fc1_W = torch.from_numpy(weights[:s_size * h_size].reshape(s_size, h_size))
        fc1_b = torch.from_numpy(weights[s_size * h_size:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end + (h_size * a_size)].reshape(h_size, a_size))
        fc2_b = torch.from_numpy(weights[fc1_end + (h_size * a_size):])
        #        print(fc2_b.shape)
        # set the weights for each layer
        # =============================================================================
        #         print("weights before copying")
        #         print(self.qnetwork_local.state_dict())
        # =============================================================================
        self.qnetwork_local.fc1.weight.data.copy_(fc1_W.view_as(self.qnetwork_local.fc1.weight.data))
        self.qnetwork_local.fc1.bias.data.copy_(fc1_b.view_as(self.qnetwork_local.fc1.bias.data))
        self.qnetwork_local.fc2.weight.data.copy_(fc2_W.view_as(self.qnetwork_local.fc2.weight.data))
        self.qnetwork_local.fc2.bias.data.copy_(fc2_b.view_as(self.qnetwork_local.fc2.bias.data))

    # =============================================================================
    #         print("weights after copying")
    #         print(self.qnetwork_local.state_dict())
    # =============================================================================

    def rmse(self, predictions, targets):
        # =============================================================================
        #         print(predictions)
        #         print("-------")
        #         print(targets)
        # =============================================================================
        predictions = np.array(predictions)
        targets = np.array(targets)
        return np.sqrt(((predictions - targets) ** 2).mean())

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        #        print("Q targets next",self.qnetwork_target(next_states).detach().max(1)[0])
        # Compute Q targetssamples  for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        #        print(dones)
        #        print("the q targets",Q_targets)

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        #        print(Q_expected.shape)

        trainsize = len(self.memory)
        samples = self.samples

        netw = self.topology

        y_train = Q_targets.numpy()

        w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]
        #        print(w_size)
        pos_w = np.ones((samples, w_size))

        pos_tau = np.ones((samples, 1))

        fxtrain_samples = np.ones((samples, trainsize, int(np.array(y_train).shape[1])))

        rmse_train = np.zeros(samples)

        w = np.random.randn(w_size)

        w_proposal = np.random.randn(w_size)

        step_w = 0.02

        step_eta = 0.01

        self.set_weights(w)

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        #        print("Q targets next",self.qnetwork_target(next_states).detach().max(1)[0])
        # Compute Q targets for current states

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        #        print(dones)
        #        print("the q targets",Q_targets)

        y_train = Q_targets.numpy()

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        #        print(Q_expected.shape)

        pred_train = Q_expected.detach().numpy()

        eta = np.log(np.var(np.array(pred_train) - np.array(y_train)))

        tau_pro = np.exp(eta)

        #        err_nn = np.sum(np.square(np.array(pred_train) - np.array(y_train)))/(len(pred_train))

        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0

        prior_likelihood = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro)  # takes care of the gradients
        [likelihood, pred_train] = self.likelihood_func(pred_train, y_train, w, tau_pro)
        #        [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.test_x,self.test_y, w, tau_pro)

        #        print(likelihood,' is likelihood of train')

        naccept = 0

        for i in range(samples - 1):
            # print(i)

            w_proposal = w + np.random.normal(0, step_w, w_size)

            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)

            self.set_weights(w_proposal)
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            #        print("Q targets next",self.qnetwork_target(next_states).detach().max(1)[0])
            # Compute Q targets for current states
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            y_train = Q_targets.numpy()

            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(states).gather(1, actions)
            #            print(Q_expected.shape)

            pred_train = Q_expected.detach().numpy()
            rmsetrain = self.rmse(pred_train, y_train)

            [likelihood_proposal, pred_train] = self.likelihood_func(pred_train, y_train, w_proposal,
                                                                     tau_pro)

            # likelihood_ignore  refers to parameter that will not be used in the alg.

            prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,
                                               tau_pro)  # takes care of the gradients

            diff_likelihood = likelihood_proposal - likelihood
            diff_priorliklihood = prior_prop - prior_likelihood
            #            print(diff_likelihood + diff_priorliklihood)

            # mh_prob = min(1, math.exp(diff_likelihood + diff_priorliklihood))
            mh_prob = min(0, (diff_likelihood + diff_priorliklihood))
            mh_prob = math.exp(mh_prob)
            #   print("prob =",mh_prob)
            u = random.uniform(0, 1)

            if u < mh_prob:
                # Update position
                # print(i, ' is the accepted sample')
                naccept += 1
                likelihood = likelihood_proposal
                prior_likelihood = prior_prop
                w = w_proposal
                eta = eta_pro
                #                print(i,rmsetrain,mh_prob)
                # =============================================================================
                #                 print("weights before sending")
                #                 print(self.qnetwork_local.state_dict())
                # =============================================================================
                self.set_weights(w)
                # =============================================================================
                #                 print("weights after sending")
                #                 print(self.qnetwork_local.state_dict())
                # =============================================================================
                #                self.soft_update(self.qnetwork_local, self.qnetwork_local, TAU)
                self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
                # if i % 100 == 0:
                #     #print ( likelihood, prior_likelihood, rmsetrain, rmsetest, w, 'accepted')
                #     print ('Sample:',i, 'RMSE train:', rmsetrain, 'RMSE test:',rmsetest)

            # =============================================================================
            #                 pos_w[i + 1,] = w_proposal
            #                 pos_tau[i + 1,] = tau_pro
            #                 fxtrain_samples[i + 1,] = pred_train
            #                 fxtest_samples[i + 1,] = pred_test
            #                 rmse_train[i + 1,] = rmsetrain
            #                 rmse_test[i + 1,] = rmsetest
            #
            #                 plt.plot(x_train, pred_train)
            # =============================================================================

            # =============================================================================
            #             else:
            #                 pos_w[i + 1,] = pos_w[i,]
            #                 pos_tau[i + 1,] = pos_tau[i,]
            #                 fxtrain_samples[i + 1,] = fxtrain_samples[i,]
            #                 fxtest_samples[i + 1,] = fxtest_samples[i,]
            #                 rmse_train[i + 1,] = rmse_train[i,]
            #                 rmse_test[i + 1,] = rmse_test[i,]
            #
            # =============================================================================
            # print i, 'rejected and retained'

            if i % 1000 == 0:
                # print ( likelihood, prior_likelihood, rmsetrain, rmsetest, w, 'accepted')
                print('Sample:', i, 'RMSE train:', rmsetrain)

        print(naccept, ' num accepted')
        print((naccept * 100) / (samples * 1.0), '% was accepted')
        accept_ratio = naccept / (samples * 1.0) * 100
        print("accept ratio=", accept_ratio)

        # Compute loss

    # =============================================================================
    #         loss = F.mse_loss(Q_expected, Q_targets)
    #         print(loss)
    #         # Minimize the loss
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    # =============================================================================

    # ------------------- update target network ------------------- #

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            # =============================================================================
            #             print("target param=",target_param)
            #             print("local param=",local_param)
            #
            # =============================================================================

            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def likelihood_func(self, fx, y, w, tausq):
        # y = data[:, self.topology[0]]
        y = y

        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(np.array(y) - np.array(fx)) / tausq
        return [np.sum(loss), fx]

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)