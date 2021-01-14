import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sc
from scipy.stats import rv_continuous

from scipy.stats import rv_continuous
from scipy.stats import hypergeom


n = 100
aa = []

# p = 0.384
# q = 0.505

p = 0.4
q = 0.6
V = p**2*q**2-q**2*p+p*q-q*p**2

print(V)
print(np.sqrt(V))
for i in range(2000):
    x = np.random.binomial(1, p, size=n)
    y = np.random.binomial(1, q, size=n)
    # y = [np.random.binomial(1, 0.2) if i == 1 else np.random.binomial(1, 0.8) for i in x]
    xy = pd.DataFrame([x, y]).T
    xy.columns = ['x', 'y']
    xy['r'] = xy.x * xy.y
    aa.append(xy.mean())

bb = pd.concat(aa, axis=1).T
diff = bb.r - bb.y * bb.x
dd = diff / diff.std()
dd.plot(kind='hist', bins=25, alpha=0.3)
ee = diff * np.sqrt(n) / np.sqrt(V)
ee.plot(kind='hist', bins=25, alpha=0.3)
plt.show()
var_v = diff.var() * n
std_v = np.sqrt(var_v)
print(var_v)
print(std_v)
quit()

n = 1000
aa = []
for i in range(10):
    x = np.random.binomial(1, 0.384, size=n)
    y = np.random.binomial(1, 0.505, size=n)
    # y = [np.random.binomial(1, 0.2) if i == 1 else np.random.binomial(1, 0.8) for i in x]
    xy = pd.DataFrame([x, y]).T
    xy.columns = ['x', 'y']
    xy['r'] = xy.x * xy.y
    aa.append(xy.mean())


print(pd.concat(aa, axis=1).T)
print(pd.concat(aa, axis=1).T.mean())
print(pd.concat(aa, axis=1).T.cov() * n)


n = 100000
x = np.random.binomial(1, 0.5, size=n)
y = [np.random.binomial(1, 0.75) if i == 1 else np.random.binomial(1, 0.25) for i in x]
xy = pd.DataFrame([x, y]).T
xy.columns = ['x', 'y']
xy['r'] = xy.x * xy.y
print(xy.mean())


n=10
x = np.random.binomial(1, 0.5, size=n)
y = [np.random.binomial(1, 0.75) if i == 1 else np.random.binomial(1, 0.25) for i in x]

# n = 100 and n = 1000000
n_it = 10000
n = 100
means = pd.Series(np.random.chisquare(n, n_it))/n
print(means.quantile(0.95), means.var(), means.mean())

n = 1000
n_it = 100000
means = pd.Series(np.random.chisquare(n, n_it))
normalized = ((means/n) - 1)*np.sqrt(n)
print(normalized.quantile(0.975))



n = 10
means = []
for i in range(10000):
    mn = (pd.Series(np.random.standard_normal(n))**2).mean()
    means.append(mn)
((pd.Series(means) - 1) * np.sqrt(n)).plot(kind='hist', bins=100)
plt.show()

print('ciao')

class gaussian_gen(rv_continuous):
    """Gaussian distribution"""
    def _pdf(self, x):
        return np.exp(-x**2 / 2.) * x

gaussian = gaussian_gen(a=0, name='gaussian')
a = pd.Series(np.random.normal(0, 1, 100))
b = pd.Series(np.random.normal(0, 1, 100))
c = ((a - a.mean())**2).mean()
d = ((a - a.median())**2).mean()
print('mean', c, 'median', d)
e = (abs(a - a.median())).mean()
f = (abs(a - a.mean())).mean()
print('mean', f, 'median', e)
g = (abs(a - a.mean())).var()
h = (abs(a - a.median())).var()
print('mean', g, 'median', h)


# Cochrane
res_dict = {}
for i in range(1000):
    a = pd.Series(np.random.normal(0, 1, 100))
    res_dict[i] = [a.mean(), a.std()]

pd.DataFrame(res_dict).T.plot(x=0, y=1, kind='scatter')
plt.show()


X = pd.DataFrame([[1, 1, 1],[1, 2, 4],[1, 3, 9],[1, 4, 16],[1, 5, 25],
                  [1, 6, 36],[1, 7, 49],[1, 8, 64], [1, 9, 81], [1, 10, 100]])

y = [1, 3, 5, 8, 11, 14, 18, 21, 25, 28]

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
X = pd.DataFrame([[1, 1, 1],[1, 2, 4],[1, 3, 9],[1, 4, 16],[1, 5, 25],
                  [1, 6, 36],[1, 7, 49],[1, 8, 64], [1, 9, 81], [1, 10, 100]])
print(X.T.dot(X))

a = pd.Series(np.random.uniform(0, 1, 100)).sort_values().reset_index().drop('index', axis=1).reset_index()
a.columns = ['theoretical', 'real']
a.theoretical = a.theoretical + 1
a.theoretical = a.theoretical/len(a.index)
a.plot.scatter('theoretical', 'real')
plt.show()

a = [0.01,
0.1,
0.2,
0.28,
0.8]

a = pd.Series([0.01,
0.1,
0.2,
0.28,
0.8]).sort_values().reset_index().drop('index', axis=1).reset_index()
a.columns = ['theoretical', 'real']
a.theoretical = a.theoretical + 1
a.theoretical = a.theoretical/len(a.index)
a.plot.scatter('theoretical', 'real')
plt.show()


# Chi squared as an estimate of variance
n = 3
dof = n - 1
trials = 10000
sigma = 2

normalized_Sn = []
for _ in range(trials):
    s = pd.Series(np.random.normal(0, 2, n))
    Sn = ((s**2).mean() - (s.mean()**2))
    normalized_Sn.append(Sn*n/(sigma**2))

fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111)

theoretical_dist = pd.Series(np.random.chisquare(dof, trials))
theory_and_practice = pd.concat([pd.Series(normalized_Sn), theoretical_dist], axis=1)
theory_and_practice.plot(kind='hist', ax=ax, bins=100, alpha=0.5)
ax.legend(['Normalized sample variance with %s samples' %str(n), 'Chi Squared with %s degree of freedom' %str(dof)])
plt.show()
quit()
l = []
n2 = 10000

mean1 = 1
mean2 = 1  # H_0 respected
sd1 = 2
sd2 = 3

for i in range(n2):
    n = 1000
    gauss1 = pd.Series(np.random.normal(mean1, sd1, n))
    gauss2 = pd.Series(np.random.normal(mean2, sd2, n))

    diff = gauss1.mean() - gauss2.mean()
    l.append(diff)

diff_distr = (pd.Series(l) * np.sqrt(n) / np.sqrt(sd1**2 + sd2**2))
diff_distr.plot(kind='hist', bins=100, alpha=0.5)
pd.Series(np.random.normal(0, 1, n2)).plot(kind='hist', bins=100, alpha=0.5)

plt.show()

fin = 0.8
l1 = []
l2 = []
for theta in np.arange(0.33, fin, 0.01):
    tau = 1.0
    p = theta ** tau
    as_var = ((theta ** (2 - 2 * tau)) / (tau ** 2)) * (p * (1 - p))
    l1.append(as_var)

    tau = 2.0
    p = theta ** tau
    as_var = ((theta ** (2 - 2 * tau)) / (tau ** 2)) * (p * (1 - p))
    l2.append(as_var)

pd.Series(l1).plot()
pd.Series(l2).plot()

d = {}
for tau in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]:
    as_var = ((theta ** (2 - 2 * tau)) / tau ** 2) * (p * (1 - p))
    d[tau] = as_var

pd.Series(d).plot()
plt.show()

# CONFIDENCE INTERVAL
n = 500
theta = 0.6
tau = 2
p = theta ** tau

l1 = []
l2 = []
for _ in range(1000):
    draws = pd.Series(np.random.binomial(1, p, n))
    l1.append(draws.mean())
    l2.append((draws.mean())**(1.0/tau))

# print(pd.Series(l2))

print(pd.Series(l1).var(), (p*(1-p))/n)
print(pd.Series(l2).var(), (((theta**(2 - 2*tau))/tau**2)*(p*(1-p)))/n)

# CONFIDENCE INTERVAL
n = 100  # number of samples
beta = 1/ 0.478035801 # in python, beta = 1/lambda. SO lambda = 2

# in the exercise, lambda = ln(1/theta), hence theta=1/exp(lambda) or
#                                              theta=1/exp(1/avg(X_i))

l = []
for _ in range(10000):
    draws = np.random.exponential(beta, n)
    l.append(1 / np.exp(1 / draws.mean()))  # theta estimates
theta_dist = pd.Series(l)
theta_mean = theta_dist.mean()
theta_asymptotic_var = (1 / (np.exp(1/beta) ** 2) * (1/beta)**2)  # known

print(theta_dist.var(), theta_asymptotic_var/float(n))

theta_dist.plot(kind='hist', bins=50)
plt.axvline(x=theta_mean + 1.65 * np.sqrt(theta_asymptotic_var/float(n)), c='red', linewidth=2)
print(theta_mean + 1.65 * np.sqrt(theta_asymptotic_var/float(n)))
plt.show()


n = 10
l = []
for _ in range(10000):
    a = np.random.exponential(0.5, n)
    l.append(1 / a.mean())

mean_s = pd.Series(l)

print(mean_s.var(), 4.0 / float(n))
### Exercise 1a
n = 100000
a = pd.Series(np.random.normal(0, 1, n))
b = pd.Series(np.random.normal(0, 1, n)) ** 5
c = pd.concat([a, b], axis=1)
c.columns = ['a', 'b']
flt_c = c[(c.a < 2) & (c.a > -2) & (c.b < 2) & (c.b > -2)]
print(float(len(flt_c.index)) / float(n))

### Exercise 1a
n = 1000
l = []
for i in range(n):
    a = pd.Series(np.random.normal(0, 1, 5))
    l.append(a.mean())

print(pd.Series(l).var(), 1.0 / float(5))(n)

### Exercise 1c
n = 100
l2 = []
for i in range(1, 10000):
    l = []
    for _ in range(n):
        a = pd.Series(np.random.normal(0, 1, i))
        l.append(a.mean())
    l2.append(pd.Series(l).sort_values().iloc[75])

pd.Series(l2).plot()
plt.show()

result = []
for i in range(1000):
    sample1 = np.random.uniform(0, 1, 100000)

    sample2 = []
    for i in sample1:
        sample2.append(np.random.uniform(i, i + 1))

    aa = pd.concat([pd.Series(sample1), pd.Series(sample2)], axis=1).cov()
    result.append(aa.iloc[0, 1])

bb = pd.Series(result)
bb.plot()
cc = pd.Series(bb.mean(), index=bb.index)
cc.plot()
print(bb.mean())
plt.show()
quit()
n = 100
sample = np.random.normal(0, 1, n)

aa = []
for i in range(100):
    n = 1000000
    sample = np.random.normal(0, 2, n)
    aa.append(pd.concat([pd.Series(sample), pd.Series(sample) ** 2], axis=1).cov().iloc[0, 1])

pd.Series(aa).plot()

p = 0.5
n = 10
var_l = []
for _ in range(1000):
    a = pd.Series(np.random.binomial(1, p, n)).mean()
    b = a * (1 - a)
    var_l.append(b)

print(pd.Series(var_l).mean() - p * (1 - p), ((p - p ** 2) / n))

p = 0.5

mean_l = []
for _ in range(1000):
    a = pd.Series(np.random.binomial(1, p, 100)).mean() ** 2
    mean_l.append(a)

c_r = [float(0.5), float(0.75), float(1), float(2)]

for c in c_r:
    exp_mean = pd.Series(np.random.binomial(1, 0.5, 1000)).expanding().mean() - float(0.5)
    exp_mean = exp_mean * (1000 ^ float(c))
    exp_mean.plot()
    plt.show()

quit()
# Final Exam Exercise 4
mean_l = []
for i in range(1, 10000):
    a = pd.Series(np.random.uniform(0, 1, i)).cumsum()
    c_n = float(i) / 2 + np.sqrt(float(i) / 12)
    mean_l.append(a.iloc[-1] > c_n)

print(mean_l)
# print(pd.Series(mean_l).expanding().mean())
pd.Series(mean_l).expanding().mean().plot()
plt.show()
quit()

# Final Exam Exercise 2
size = 100000
gloves = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10]
l = []

for i in range(size):
    aa = np.random.choice(gloves, 8, replace=False)
    if not len(aa) == len(set(aa)):
        l.append(1)
    else:
        l.append(0)

print(float(pd.Series(l).sum()) / float(size))
quit()

# Final Exam Exercise 1
size = 1000000
a = np.random.geometric(0.5, size)
b = np.random.geometric(0.5, size)
ab = pd.DataFrame([a, b]).T

equal = ab[0] == ab[1]
print(equal.sum())

min_df = ab.min(axis=1)
print(1 - float((min_df <= 4).sum()) / float(size))
quit()

# Final Exam Exercise 5
size = float(10000)
aa = pd.Series(np.random.poisson(6, size))
print(float(len(aa[aa == 9].index)) / size)
print('ciao')

# Final Exam Exercise 3
size = 100
a = np.random.uniform(0, 1, size)
b = np.random.uniform(0, 1, size)
ab = pd.DataFrame([a, b]).T
print(ab.min(axis=1).mean())
print('ciao')

##############

mean_l = []
conf_int_up = []
conf_int_down = []

for i in range(10000):
    a = np.random.binomial(i, 0.3333) - (0.3333 * i)
    mean_l.append(a)
    conf_int_down.append(30 / np.sqrt(2 * i))
    conf_int_up.append(-30 / np.sqrt(2 * i))

print(mean_l)
pd.Series(mean_l).plot(ax=ax1)
pd.Series(conf_int_down).plot(ax=ax1)
pd.Series(conf_int_up).plot(ax=ax1)
plt.show()
quit()

print(pd.Series(mean_l).var())
quit()
mean_l = []
for i in range(100000):
    a = np.random.binomial(100, 0.4)
    b = np.random.binomial(100, 0.5)
    c = np.random.binomial(100, 0.6)
    mean = a + b + c
    mean_l.append(mean)

print(pd.Series(mean_l).var())
quit()

a = pd.Series(np.random.normal(2, 3, 100000))
b = pd.concat([a * 1 / 3, a.shift() * 2 / 3], axis=1).dropna().sum(axis=1).expanding().mean()
b.plot()
plt.show()
print(b)
quit()

draws = []
for n in range(1, 1000):
    # a = np.random.exponential(1/float(n))
    a = np.random.exponential(n)
    print(a)
    draws.append(a)

pd.Series(draws).plot()
plt.show()
quit()

## Exercise 1
a = 5

rnd_list = list(np.random.uniform(0, a, 100000))
aa = [int(i) for i in rnd_list]
bb = pd.Series(aa).groupby(by=aa).count()
cc = pd.concat([pd.Series(rnd_list), pd.Series(aa)], axis=1)
res = cc.iloc[:, 0] - cc.iloc[:, 1]
res.plot(kind='hist', bins=50)
plt.show()
print(bb)
quit()

## Exercise 2
h_l = []
for i in range(100000):
    x = np.random.uniform(0, 1)
    y = np.random.uniform(0, 1)
    h = (x + 2) * y
    h_l.append(h)

aa = pd.Series(np.log(h_l))
bb = aa[aa < np.log(2)]
print(len(bb[bb < 0].index) / float(len(bb.index)))

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

pd.Series(h_l).plot(ax=ax1, kind='hist', bins=100)
pd.Series(np.log(h_l)).plot(ax=ax2, kind='hist', bins=100)

plt.show()
quit()

# excercise 6

ratio_l = []
for i in range(100000):
    broken = np.random.uniform(0, 0.5)
    other_piece = 1 - broken
    ratio_l.append(broken / other_piece)

print(pd.Series(ratio_l).mean())
quit()
pd.Series(ratio_l).plot(kind='hist', bins=100)
plt.show()
quit()
small_l = []
big_l = []
for i in range(100000):
    broken = np.random.uniform(0, 1)
    other_piece = 1 - broken

    small = min([broken, other_piece])
    big = max([broken, other_piece])
    small_l.append(small)
    big_l.append(big)

# excercise 5
original_signal_l = []
signal_l = []

for i in range(100000):
    original_signal = np.random.choice([-1, 1])
    noise = np.random.normal(0, 1)
    signal = noise + original_signal

    original_signal_l.append(original_signal)
    signal_l.append(signal)

stuff = pd.concat([pd.Series(original_signal_l), pd.Series(signal_l)], axis=1)
stuff.columns = ['os', 's']
flt_stuff = stuff[(stuff.s > 0.18) & (stuff.s < 0.22)]
print(float(len(flt_stuff[flt_stuff.os == 1].index)) / float(len(flt_stuff.index)))

quit()

## Exercise 2b
dist = []
for i in range(100000):
    a = np.random.exponential()
    sc.expon.cdf(a)
    dist.append(sc.expon.cdf(a))

pd.Series(dist).plot(kind='hist')
plt.show()
quit()

pd.Series(small_l).plot(kind='hist', bins=10)
plt.show()
aa = pd.concat([pd.Series(small_l), pd.Series(big_l)], axis=1)
aa.columns = ['short', 'long']
(aa.short / aa.long).plot(kind='hist', bins=100)
plt.show()

print('ciao')
print('ciao')
print('ciao')

print('ciao')

## Exercise 4
std_n_l = []
y_l = []
for i in range(100000):
    std_n = np.random.normal(0, 1)
    y = np.random.normal(-2 * std_n, 1)
    std_n_l.append(std_n)
    y_l.append(y)

aa = pd.concat([pd.Series(std_n_l), pd.Series(y_l)], axis=1)
aa.columns = ['std_n', 'y']

print(aa.var())
cond = aa[(aa.y > -1.05 + 1.5) & (aa.y < -0.95 + 1.5)]
print('cond mean of y', cond.mean().loc['y'])
print('cond mean of x', cond.mean().loc['std_n'])
print('cond var of x', cond.var().loc['std_n'])
quit()


def exercise_5():
    # === ex. 5
    cnt = 0
    cnt2 = 0
    for j in range(100000):
        kk = random.choice([1, 2, 3, 4])
        choices = []

        for i in range(kk):
            choices.extend([random.choice([1, 2, 3, 4, 5, 6])])

        aa = sum([1 if i in [5, 6] else 0 for i in choices])
        # print(choices, aa)
        if aa == 2:
            cnt += 1
            if kk == 3:
                cnt2 += 1
                # print(cnt, j)

    print(float(cnt) / float(j))
    print('---')
    print(float(cnt2) / float(cnt))
    quit()
    # == ex 2
    len_c = []
    for i in range(100000):
        choices = []

        for _ in range(6):
            choices.extend([random.choice([1, 2, 3])])

        len_c.append(len(set(choices)))

    aa = pd.Series(len_c)
    print(1 - (float(len(aa[aa == 3].index) / float(i))))


def exercise_2_part_1():
    # EXERCISE 1 part 1
    cnt = 0
    for i in range(1, 10001):
        l = list(np.arange(1, 21))
        r_sample = []
        for j in range(10):
            r_sample.append(random.choice(l))

        aa = len(r_sample) - len(set(r_sample))
        print(aa)
        if aa > 0:
            cnt += 1

    print(float(cnt) / float(i))


def exercise_2_part_2():
    # EXERCISE 1 part 2
    s = list(np.arange(1, 21))

    cnt = 0
    for i in range(1, 100001):
        aa = pd.Series(random.sample(s, 4)).sort_values()
        cnt += aa.iloc[1] == 7

    print(float(cnt) / float(i))


exercise_2_part_2()
