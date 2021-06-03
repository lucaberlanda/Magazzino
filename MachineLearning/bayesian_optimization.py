# example of bayesian optimization for a 1d function from scratch
from math import sin
from math import pi
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy import sort
from numpy.random import normal
from numpy.random import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot

pyplot.rcParams["figure.figsize"] = (10, 6)


# objective function
def objective_unimodal(x, noise=0.1):
    noise = normal(loc=0, scale=noise)
    return (x ** 2 * sin(5 * pi * x) ** 6.0) + noise


def objective(x, noise=1.5):
    noise = normal(loc=0, scale=noise)
    return sin(10 * x) + noise


def surrogate(model, X):
    # catch any warning generated when making a prediction
    with catch_warnings():
        # ignore generated warnings
        simplefilter("ignore")
        return model.predict(X, return_std=True)


# plot real observations vs surrogate function
def plot(X, y, model, title=''):
    pyplot.scatter(X, y, color='darkblue', alpha=0.5)
    Xsamples = asarray(arange(0, 1, 0.001))
    Xsamples = Xsamples.reshape(len(Xsamples), 1)
    ysamples, _ = surrogate(model, Xsamples)
    pyplot.plot(Xsamples, ysamples, color='green')
    pyplot.title(title, size=15)
    pyplot.show()


# optimize the acquisition function
def opt_acquisition(X, model):
    Xsamples = random(100)
    Xsamples = Xsamples.reshape(len(Xsamples), 1)

    yhat, _ = surrogate(model, X)  # calculate the acquisition function for each sample
    best = max(yhat)

    mu, std = surrogate(model, Xsamples)  # calculate mean and stdev via surrogate function
    print(mu, std)
    mu = mu[:, 0]

    scores = norm.cdf((mu - best) / (std + 1E-9))  # calculate the probability of improvement
    ix = argmax(scores)

    return Xsamples[ix, 0]


X = random(100)
y = asarray([objective(x) for x in X])
X = X.reshape(len(X), 1)
y = y.reshape(len(y), 1)
model = GaussianProcessRegressor()
model.fit(X, y)

# plot original function and noisy data
X = sort(random(100))
y = asarray([objective(x) for x in X])
y_no_noise = [objective(x, noise=0) for x in X]
pyplot.scatter(sort(X), y, color='darkblue', alpha=0.5)
pyplot.plot(sort(X), y_no_noise, color='darkblue')
pyplot.title('Original Function + Noisy Data', size=15)
pyplot.show()

# plot before hand
# sample the domain sparsely with noise
X = X.reshape(len(X), 1)
y = y.reshape(len(y), 1)

model = GaussianProcessRegressor()
model.fit(X, y)

plot(X, y, model, title='Noisy Data and Estimated Gaussian Process')
pyplot.show()

# perform the optimization process
for _ in range(50):
    # select the next point to sample
    x = opt_acquisition(X, model)
    # sample the point
    actual = objective(x)
    # summarize the finding
    est, _ = surrogate(model, [[x]])
    # print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))

    # add the data to the dataset
    X = vstack((X, [[x]]))
    y = vstack((y, [[actual]]))

    # update the model
    model.fit(X, y)
    plot(X, y, model)

# plot all samples and the final surrogate function
plot(X, y, model)
# best result
ix = argmax(y)
print('Best Result: x=%.3f, y=%.3f' % (X[ix], y[ix]))
