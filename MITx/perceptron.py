import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_1 = np.array([-1, 0, 0, 0])
x_2 = np.array([0, 1, 0, 0])
x_3 = np.array([0, 0, -1, 0])
x_4 = np.array([0, 0, 0, 1])

y_1 = 1
y_2 = 1
y_3 = 1
y_4 = 1

xs = [x_1, x_4, x_3, x_2]
ys = [y_1, y_4, y_3, y_2]

scatter = False
if scatter:
    fig1, ax1 = plt.subplots()
    xs_df = pd.DataFrame(xs)
    xs_df.columns = ['x', 'y']
    xs_df.iloc[:2, :].plot(kind='scatter', x='x', y='y', ax=ax1)
    xs_df.iloc[2:, :].plot(kind='scatter', x='x', y='y', ax=ax1, c='red')
    plt.show()

theta = np.array([0, 0, 0, 0])  # initialize theta

T = 100
error_cnt = 0
for t in np.arange(0, T):
    for i, x in enumerate(xs):
        print(theta)
        if ys[i] * (np.dot(theta, x)) <= 0:
            error_cnt += 1
            print(i)
            print('error', error_cnt)
            # print(to_use_y[i], x, theta, (np.dot(theta, x)))
            theta = theta + x * ys[i]
quit()
theta_0 = 0
T = 100
error_cnt = 0
for t in np.arange(0, T):
    for i, x in enumerate(xs):
        print(theta, theta_0)
        if ys[i] * (np.dot(theta, x) + theta_0) <= 0:
            error_cnt += 1
            print(i)
            print('error', error_cnt)
            # print(to_use_y[i], x, theta, (np.dot(theta, x)))
            theta = theta + x * ys[i]
            theta_0 = theta_0 + ys[i]

quit()

x_1 = np.array([-1, -1])
x_2 = np.array([1, 0])
x_3 = np.array([-1, 10])
y_1 = 1
y_2 = -1
y_3 = 1

first_x1_x = [x_1, x_2, x_3]
first_x2_x = [x_2, x_3, x_1]
first_x1_y = [y_1, y_2, y_3]
first_x2_y = [y_2, y_3, y_1]

to_use_x = first_x1_x
to_use_y = first_x1_y

theta = np.array([0, 0])  # initialize theta
theta_0 = 0

T = 100
error_cnt = 0
for t in np.arange(0, T):
    for i, x in enumerate(to_use_x):
        print(theta)
        if to_use_y[i] * (np.dot(theta, x)) <= 0:
            error_cnt += 1
            print('error', error_cnt)
            # print(to_use_y[i], x, theta, (np.dot(theta, x)))
            theta = theta + x * to_use_y[i]
