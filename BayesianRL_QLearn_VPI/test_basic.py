import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from theano import shared
import theano
# obs_y = np.random.normal(0.5,0.35,2000)
#
# with pm.Model() as excercise:
#     stdev = pm.HalfNormal('stdev',sd=1.)
#     mu = pm.Normal('mu',mu=0.0,sd=1.0)
#
#     # monte-carlo
#     y = pm.Normal('y',mu=mu,sd=stdev,observed=obs_y)
#
#     trace = pm.sample(1000,cores=1,return_inferencedata=True)
#     pm.traceplot(trace,['mu','stdev'])
#     plt.show()

# chapter 2 Linear Regression

# N = 10000
# noise = np.random.normal(0.0,0.1,N)
#
# X = np.random.normal(1.0,0.1,N)
#
# obs_y = (0.65*X) + 0.5 + noise
#
# with pm.Model() as ex2:
#     stdev = pm.HalfNormal('stdev',sd=1.)
#     intercept = pm.Normal('intercept',mu=0.0,sd=1.)
#     coeff = pm.Normal('beta', mu=0.5, sd=1.)
#     expected_val = (X*coeff) + intercept
#     y = pm.Normal('y',mu=expected_val,sd=stdev, observed=obs_y)
#     trace = pm.sample(1000,cores=1,return_inferencedata=True)
#     pm.traceplot(trace, ['intercept','beta', 'stdev'])
#     plt.show()
#
# # predictions
# with ex2:
#     ppc = pm.sample_posterior_predictive(trace,samples=1000)
#     y_preds = ppc['y']
#     print('y_pred shape ',y_preds.shape)
#
#     expected_y_pred = np.reshape(np.mean(y_preds,axis=0), [-1])
#     plt.scatter(X,expected_y_pred, c='g')
#     plt.scatter(X,obs_y,c='b',alpha=.1)
#     plt.title('Relationship between X and pred Y')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.show()

# chapter 3 Logistic Regression

# # 1. generate the artificial dataset.
# N = 10000
#
# X = np.random.uniform(0, 1, N)
# def DGP(x):
#     obs_y = []
#     for n in range(len(x)):
#         if x[n] > (0.7 + np.random.normal(0.0, 0.0001, 1)[0]):
#             obs_y.append(1.0)
#         else:
#             obs_y.append(0.0)
#
#     return obs_y
#
# X_shared = shared(X)
# obs_y = DGP(X)
#
# # 2. model that data with a simple regression model
# with pm.Model() as exercise3:
#
#     intercept = pm.Normal('intercept', mu=0.0, sd=.1)
#     coeff = pm.Normal('beta', mu=0.0, sd=.1)
#
#     expected_value = pm.math.invlogit((coeff * X_shared) + intercept)
#     y = pm.Bernoulli('y', expected_value, observed=obs_y)
#
#     trace = pm.sample(1000,cores=1,return_inferencedata=True)
#
#     pm.traceplot(trace, ['intercept', 'beta'])
#     plt.show()
#
# # 3. posterior predictive checks
# TEST_N = 1000
# testX = np.random.uniform(0, 1, TEST_N)
# testY = DGP(testX)
#
# X_shared.set_value(testX)
#
# ppc = pm.sample_posterior_predictive(trace, model=exercise3, samples=500)
# y_preds = ppc['y']
#
# print("y_preds shape = ", y_preds.shape)
#
# expected_y_pred = np.reshape(np.mean(y_preds, axis=0), [-1])
#
# plt.scatter(testX, expected_y_pred, c='g')
# plt.scatter(testX, testY, c='b', alpha=0.1)
# plt.title("Relationship between X and (predicted) Y")
# plt.xlabel("X")
# plt.ylabel("Y")
#
# plt.show()

# Bayesian Neural Network
# 1. generate a simple non-linear function
X = np.reshape(np.arange(-5.0, 5.0, 0.01), [-1, 1])

print ("shape of X: ", X.shape)

Y = X ** 2.0

plt.scatter(X, Y)
plt.show()

# 2. neural network
print ("Building neural network...")

ann_input = theano.shared(X)
ann_output = theano.shared(Y)

n_hidden = 5

# Initialize random weights between each layer
init_1 = np.random.randn(X.shape[1], n_hidden).astype(theano.config.floatX)
init_2 = np.random.randn(n_hidden, n_hidden).astype(theano.config.floatX)
init_out = np.random.randn(n_hidden, 1).astype(theano.config.floatX)

SD = 10.
with pm.Model() as neural_network:

    # Weights from input to hidden layer
    weights_1 = pm.Normal('layer1', mu=0, sd=SD, shape=(X.shape[1], n_hidden), testval=init_1)
    bias_1 = pm.Normal('bias1', mu=0, sd=SD, shape = n_hidden)
    weights_2 = pm.Normal('layer2', mu=0, sd=SD, shape=(n_hidden, n_hidden), testval=init_2)
    bias_2 = pm.Normal('bias2', mu=0, sd=SD, shape=n_hidden)
    weights_out = pm.Normal('out', mu=0, sd=SD, shape=(n_hidden, 1), testval=init_out)
    intercept = pm.Normal('intercept', mu=0, sd=SD)

    # Now assemble the neural network
    layer_1 = pm.math.tanh(pm.math.dot(ann_input, weights_1) + bias_1)
    layer_2 = pm.math.tanh(pm.math.dot(layer_1, weights_2) + bias_2)
    layer_out = pm.math.dot(layer_2, weights_out)

    y = pm.Normal('y', layer_out + intercept, observed=ann_output)

print ("Done. Sampling...")

num_samples = 200
with neural_network:

    trace = pm.sample(num_samples, tune=num_samples, target_accept=0.999, cores=1,return_inferencedata=True)

# for each of the num_samples parameter values sampled above, sample 500 times the expected y value.
samples = pm.sample_posterior_predictive(trace, model=neural_network, size=500)

y_preds = np.reshape(samples['y'], [num_samples, 500, X.shape[0]])

# get the average, since we're interested in plotting the expectation.
y_preds = np.mean(y_preds, axis=1)
y_preds = np.mean(y_preds, axis=0)

RMSD = np.sqrt(np.mean((y_preds - Y) ** 2.0))

plt.scatter(X, Y)
plt.scatter(X, y_preds, alpha=0.1)
plt.show()