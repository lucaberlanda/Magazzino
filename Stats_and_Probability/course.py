import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sc

# Final Exam Exercise 4
mean_l = []
for i in range(1, 10000):
    a = pd.Series(np.random.uniform(0,1, i)).cumsum()
    c_n = float(i)/2 + np.sqrt(float(i) / 12)
    mean_l.append(a.iloc[-1]> c_n)

print(mean_l)
#print(pd.Series(mean_l).expanding().mean())
pd.Series(mean_l).expanding().mean().plot()
plt.show()
quit()

# Final Exam Exercise 2
size=100000
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

fig = plt.figure()
ax1 = fig.add_subplot(111)
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

print('ciao')
quit()

a = pd.Series(np.random.normal(2, 3, 100000))
b = pd.concat([a * 1/3, a.shift() * 2/3], axis=1).dropna().sum(axis=1).expanding().mean()
b.plot()
plt.show()
print(b)
quit()


draws = []
for n in range(1, 1000):
    #a = np.random.exponential(1/float(n))
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
    h =(x + 2) * y
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
aa = pd.concat([pd.Series(small_l),pd.Series(big_l)], axis=1)
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
    y = np.random.normal(-2*std_n, 1)
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
                cnt2 +=1
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
    print(1- (float(len(aa[aa == 3].index) / float(i))))

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