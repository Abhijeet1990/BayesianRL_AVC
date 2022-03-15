# -*- coding: utf-8 -*-
"""
Code adopted from Shubham Subhnil. The algorithm is referred from [see GitHub readme.md]
Please use the repository link and Author's name for presenting the code in academic and scientific works.
"""


# BAC Agent

import numpy as np
import math
from scipy.sparse import csr_matrix, linalg, hstack, vstack, identity
import pandas as pd

class BAC_Agent:

    def __init__(self, gridenv, learning_params):
        self.env = gridenv
        self.learning_params = learning_params
        self.data = np.zeros((self.learning_params.num_update_max, 7))
        self.grad_store = []
        self.policy_store = []

    def BAC_Function(self):
        learning_params = self.learning_params
        num_output = (learning_params.num_update_max / learning_params.sample_interval)
        perf = np.zeros((math.ceil(num_output), 3))
        evalpoint = 0

        Pandas_dataframe = pd.DataFrame(np.zeros((learning_params.num_update_max, 6)))
        Pandas_dataframe = Pandas_dataframe.astype('object')

        STEP = 1

        # Training the agent
        for i in range(0, learning_params.num_trial):

            theta = np.zeros((self.env.num_policy_param, 1))
            alpha_schedule = learning_params.alp_init_BAC * (learning_params.alp_update_param /
                                                             (learning_params.alp_update_param +
                                                              (np.arange(1, (learning_params.num_update_max + 1)) - 1)))

            for j in range(0, learning_params.num_update_max):
                batch_runtime = 0
                reward1 = 0  # Gym Reward
                reward2 = 0  # User defined reward
                # Policy evaluation after every n(sample_interval) policy updates
                if (j % (learning_params.sample_interval)) == 0:
                    perf[evalpoint, 0], perf[evalpoint, 1], perf[evalpoint, 2] = self.env.perf_eval(theta, learning_params)
                    evalpoint += 1

                # fisher information matrix
                G = csr_matrix((self.env.num_policy_param, self.env.num_policy_param), dtype=np.float16)

                # Run num_episode episodes for BAC Gradient evaluation
                # Gradient evaluation occurs in batches of episodes (e.g. 5)
                for l in range(1, learning_params.num_episode + 1):
                    t = 0
                    episode_states = []
                    episode_scores = []

                    env_current_state = self.env.reset()  # state = d.random_state(d)
                    state = list(env_current_state)
                    done = False

                    # The problem now is handling of state in calc_score.
                    # calc_score uses y array which is essentially a C map of
                    # state = (position, velocity)
                    # C maps are exclusive of each environment and observations
                    a, scr = self.env.calc_score(theta,state)
                    scr = csr_matrix(scr)

                    while done == False or t < learning_params.episode_len_max:
                        for istep in range(0, STEP):
                            if done == False:
                                # state, _ = d.dynamics(state, a, d)
                                # state = d.is_goal(state, d)
                                state, reward, done, _ = self.env.step(a)
                                state = list(state)
                                state.append(done)
                                reward1 += reward
                                reward2 -= 1

                        G = G + (scr @ csr_matrix.transpose(scr))  ## Use @ for dot multiplication...
                        ## of sparse matrices

                        episode_states.append(state)
                        episode_scores.append(scr)
                        a, scr = self.env.calc_score(theta, state)
                        scr = csr_matrix(scr)
                        t = t + 1

                    # Create the batch data of num_episode episodes
                    # to be given for gradient estimation
                    episodes = (episode_states, episode_scores, t)
                    batch_runtime += t

                # Fix the identity matrix (Fisher Information Matrix)
                G = G + 1e-6 * identity(G.shape[0])
                grad_BAC = self.BAC_Gradient(episodes, G, self.env, learning_params)

                if learning_params.alp_schedule:
                    alp = alpha_schedule[j]
                else:
                    alp = learning_params.alp_init_BAC

                avg_episode_length = batch_runtime / learning_params.num_episode
                theta = theta + (alp * grad_BAC)
                #error = state[0] - self.env.observation_space.high[0]
                error = state[0] - 1.0
                mae = abs(error) / (j + 1)
                mse = math.pow(error, 2) / (j + 1)
                # Data storing in self.data
                self.data[j] = np.array([j + 1, mae, mse, alp, reward1, reward2, avg_episode_length])
                self.grad_store.append(grad_BAC)
                self.policy_store.append(theta)

                print("Completed updates:", j + 1, "/", learning_params.num_update_max)

            Pandas_dataframe = pd.DataFrame({"Episode Batch": self.data[:, 0],
                                             "Learning Rate": self.data[:, 3],
                                             "Mean Absolute Error": self.data[:, 1],
                                             "Mean Squared Error": self.data[:, 2],
                                             "Batch Gym Reward": self.data[:, 4],
                                             "Batch User Reward": self.data[:, 5],
                                             "Avg. Episode Length (t)": self.data[:, 6],
                                             "BAC Gradient": self.grad_store,
                                             "Policy Evolution": self.policy_store})
            perf_dataframe = pd.DataFrame({"BAC Evaluation Batch": perf[:, 0], "Gym Batch Reward": perf[:, 1],
                                           "User Defined Batch Reward": perf[:, 2]})
        return perf_dataframe, theta, Pandas_dataframe


    def BAC_Gradient(self,episodes, G, domain_params, learning_params):
        gam = learning_params.gam
        gam2 = gam ** 2

        nu = 0.1
        sig = 3
        sig2 = sig ** 2
        ck_xa = 1
        sigk_x = 1.3 * 0.25

        # Initialization
        T = 0
        m = 0
        statedic = []
        scr_dic = []
        invG_scr_dic = []
        alpha = np.array(None, dtype=np.float16)
        C = np.array(None, dtype=np.float16)
        Kinv = np.array(None, dtype=np.float16)
        k = np.array(None, dtype=np.float16)
        z = np.array(None, dtype=np.float16)

        for l in range(0, learning_params.num_episode):
            ISGOAL = 0
            t = 0
            T = T + 1
            c = np.zeros((m, 1))
            d = 0
            s = math.inf
            # state and scr are lists appended as a 1-D column-wise array objects.
            state = episodes[0][t]
            scr = episodes[1][t]
            scrT = csr_matrix.transpose(scr)

            temp1 = domain_params.kernel_kxx()
            invG_scr = linalg.spsolve(G, scr)
            invG_scr = vstack(invG_scr)
            temp2 = ck_xa * (scrT @ invG_scr)
            kk = temp1 + temp2.toarray()  # kk is always a scalar but returned as 1x1 array
            # print("kk:", kk)

            if m > 0:
                k = ck_xa * (scrT @ hstack(invG_scr_dic[:]))  # State-action kernel -- Fisher Information Kernel
                k = np.transpose(k.toarray() + domain_params.kernel_kx(state, statedic))
                a = np.dot(Kinv, k)
                goop = np.dot(np.transpose(k), a)
                # print("goop:", goop)
                delta = kk - goop  # delta should be a 'scalar'
                # print("Here 1 a:", a)
                # print("delta:", delta)

            else:
                k = np.array(None, dtype=np.float16)
                a = np.array(None, dtype=np.float16)
                delta = kk
            # delta cocmes out to be a 1x1 array which must be changed to scalar
            # hence we use delta[0] and kk[0]
            if m == 0 or delta.item() > nu:
                a_hat = a
                h = np.vstack([a, -gam])  # h = [[a], [-gam]]
                if np.isnan(h[0][0]):
                    h = h[1:]

                # a = [[z], [1]]
                a = np.vstack([z, 1])
                if np.isnan(a[0][0]):
                    a = a[1:]

                # alpha = [[alpha], [0]]
                alpha = np.vstack([alpha, 0])
                if np.isnan(alpha[0][0]):
                    alpha = alpha[1:]

                # C = [[C, z], [np.transpose(z), 0]]
                C = np.block([[C, z], [np.transpose(z), 0]])
                if np.isnan(C[0][0]):
                    C = C[1:, 1:]

                # Kinv = [(delta * Kinv) + (a_hat * a_hat'), -a_hat;
                #         -a_hat'                          , 1] / delta
                Kinv = (1 / delta.item()) * np.block([[(delta.item() * Kinv) + np.dot(a_hat, a_hat.T), (-1 * a_hat)],
                                                      [(-1 * a_hat.T), 1]
                                                      ])
                if np.isnan(Kinv[0][0]):
                    Kinv = Kinv[1:, 1:]
                # print("Kinv 1:", Kinv)

                # z = [[z], [0]]
                z = np.vstack([z, 0])
                if np.isnan(z[0][0]):
                    z = z[1:]

                # c = [[c], [0]]
                c = np.vstack([c, 0])
                if np.isnan(c[0][0]):
                    c = c[1:]

                statedic.append(state)
                scr_dic.append(scr)
                invG_scr_dic.append(invG_scr)
                m = m + 1

                # k = [[k], [kk]]
                k = np.vstack([k, kk.item()])
                if np.isnan(k[0][0]):
                    k = k[1:]

            # Time-loop
            while (t < episodes[2]):
                state_old = state
                k_old = k
                if np.isnan(k_old[0][0]) and len(k) != 1:
                    k_old = k_old[1:]
                kk_old = kk.item()
                a_old = a
                if np.isnan(a_old[0][0]) and np.shape(a_old) != (1,):
                    a_old = a_old[1:]
                c_old = c
                s_old = s
                d_old = d

                r = domain_params.calc_reward(state_old[0])

                coef = (gam * sig2) / s_old

                # Goal update
                if ISGOAL == 1:
                    dk = k_old
                    dkk = kk_old
                    h = a_old
                    c = (coef * c_old) + h - np.dot(np.atleast_2d(C), dk)
                    s = sig2 - (gam * sig2 * coef) + np.dot(dk.T, c + (coef * c_old))
                    d = (coef * d_old) + r - np.dot(dk.T, np.atleast_2d(alpha))

                # Non-goal update
                else:
                    state = episodes[0][t + 1]
                    scr = episodes[1][t + 1]
                    scrT = csr_matrix.transpose(scr)

                    if state[1] == True:
                        ISGOAL = 1
                        t = t - 1
                        T = T - 1

                    temp1 = domain_params.kernel_kxx()
                    invG_scr = linalg.spsolve(G, scr)
                    invG_scr = vstack(invG_scr)
                    temp2 = ck_xa * (scrT @ invG_scr)
                    kk = temp1 + temp2.toarray()  # kk is always a 'scalar'
                    k = ck_xa * (scrT @ hstack(invG_scr_dic[:]))

                    # Looping over elements of k and kerne_kx
                    # Cannot directly add scalar and sparse matrix
                    k = k.toarray() + domain_params.kernel_kx(state, statedic)
                    k = np.transpose(k)
                    a = np.dot(Kinv, k)
                    delta = kk - np.dot(np.transpose(k), a)  # delta should be a 'scalar'

                    dk = k_old - (gam * k)
                    d = (coef * d_old) + r - np.dot(dk.T, np.atleast_2d(alpha))

                    if delta.item() > nu:
                        h = np.vstack((a_old, -gam))
                        dkk = np.dot(np.transpose(a_old), (k_old - (2 * gam * k))) + (gam2 * kk.item())
                        c = (coef * np.vstack((c_old, 0))) + h - np.vstack((np.dot(C, dk), 0))
                        arbi = np.dot(np.atleast_2d(C), dk)
                        s = ((1 + gam2) * sig2) + dkk - np.dot(dk.T, arbi) + (
                                2 * coef * np.dot(c_old.T, dk)) - (gam * sig2 * coef)

                        alpha = np.vstack([alpha, 0])
                        # C = [[C, z], [np.transpose(z), 0]]

                        C = np.block([[C, z], [np.transpose(z), 0]])
                        if np.isnan(C[0][0]):
                            C = C[1:, 1:]

                        statedic.append(state)
                        scr_dic.append(scr)
                        invG_scr_dic.append(invG_scr)

                        m = m + 1
                        # Kinv = (1/delta[0]) * [[(delta[0] * Kinv) + (a * np.transpose(a)), -1 * a],
                        #         [np.transpose(-1 * a)                      , 1]]
                        Kinv = (1 / delta.item()) * np.block([[(delta.item() * Kinv) + np.dot(a, a.T), (-1 * a)],
                                                              [(-1 * a.T), 1]
                                                              ])
                        # print("Kinv 2:", Kinv)

                        a = np.vstack([z, 1])
                        z = np.vstack([z, 0])  # [[z], [0]]
                        k = np.vstack([k, kk.item()])  # [[k], [kk]]

                    else:  # delta <= nu
                        if np.isnan(a[0][0]):
                            a = a[1:]
                        h = a_old - (gam * a)
                        try:
                            dkk = np.dot(np.transpose(h), dk)
                        except Exception as e:
                            dkk = np.dot(h, dk)

                        prod1 = np.atleast_2d(coef * c_old)
                        # if len(prod1) == 0:
                        #     prod1 = np.zeros(np.shape(h))

                        # print(C, dk)
                        prod2 = np.dot(np.atleast_2d(C), dk)
                        c = prod1 + h - prod2
                        s = np.dot(np.transpose(dk), c + prod1) + ((1 - gam2) * sig2) - (
                                gam * sig2 * coef)

                # Alpha update
                alpha = alpha + c * (d.item() / s.item())
                # C update
                C = C + np.matmul(c, np.transpose(c) / s.item())

                # Update time counters
                t = t + 1
                T = T + 1

        # For all the fuss we went through, FINALLY!
        grad = ck_xa * (hstack(scr_dic) @ alpha)

        return grad


class learning_parameters(object):
    """
    A Class for easier handling of Learning Parameters
    Another alternative is to have a "list" object but it will take further
    processing to extract variables from that list. This Class must be standard for all
    BAC environments.
    """

    def __init__(self):
        self.episode_len_max = 2  ## Length of each episode in sec (e.g. 1000, 3000).
        self.num_update_max = 200  ## Apply the policy update after 500 cycles of BAC (e.g. 25000)
        self.sample_interval = 25  ## Policy evaluation after every 25 policy updates (e.g. 1000)
        self.num_trial = 1  ## Number of times the entire experiment is run

        self.gam = 0.99  ## Discount Factor (see BAC_grad for implementation)
        self.num_episode = 5  ## Number of episodes for BAC_Gradient estimation
        # Gradient Estimate occurs in batches of episodes
        # Can use 5, 10, 15, 20... 40. This has minimal effect on the convergence.

        self.alp_init_BAC = 0.025  ## Initial learning rate
        self.alp_variance_adaptive = 0  ## Fixed variance. Change to 1 for adaptive variance
        self.alp_schedule = 0  ## Fixed learning rate. Change to 1 for adaptive 'alpha'
        self.alp_update_param = 500  ## Total number of policy updates

        self.SHOW_EVERY_RENDER = 100
        self.SIGMA_INIT = 1
