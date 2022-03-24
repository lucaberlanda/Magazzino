"""
How to gamble with demons (and make money doing it)
A deep dive into safe haven investing and managing your risk

Raposa • March 7, 2022

https://raposa.trade/blog/how-to-gamble-with-demons-and-make-money-doing-it/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def NietszcheDice(cash: float = 0,
                  returns: list = [0.5, 1.05, 1.05, 1.05, 1.05, 1.5],
                  rolls: int = 300, samples: int = 1000):
    bet = 1 - cash
    adj_returns = cash + bet * np.asarray(returns)
    roll_sims = np.random.choice(adj_returns,
                                 size=(rolls, samples)).reshape(-1, rolls)
    return roll_sims.cumprod(axis=1)


n_traj = NietszcheDice()


def getQuantilePath(trajectories: np.array, q: float = 0.5):
    quantile = np.quantile(trajectories[:, -1], q=q)
    path = trajectories[np.abs(quantile - trajectories[:, -1]).argmin()]
    return quantile, path


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_dice_game(n_traj):
    perc50, path50 = getQuantilePath(n_traj)
    perc95, path95 = getQuantilePath(n_traj, q=0.95)
    perc5, path5 = getQuantilePath(n_traj, q=0.05)
    path_avg = n_traj.mean(axis=0)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=(3, 1))
    ax = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])

    ax.plot(path50, label='Median')
    ax.plot(path95, label=r'$95^{th}$ Percentile')
    ax.plot(path5, label=r'$5^{th}$ Percentile')
    ax.plot(path_avg, label='Mean', linestyle=':')
    ax.fill_between(np.arange(n_traj.shape[1]),
                    y1=n_traj.min(axis=0),
                    y2=n_traj.max(axis=0),
                    alpha=0.3, color=colors[4])
    ax.set_title('Playing Dice with your Wealth')
    ax.set_xlabel('Rolls')
    ax.set_ylabel('Ending Wealth')
    ax.semilogy()
    ax.legend(loc=3)

    growth = (np.power(n_traj[:, -1], 1 / 300) - 1) * 100
    growth_med = (np.power(path50[-1], 1 / 300) - 1) * 100
    growth_avg = (np.power(path_avg[-1], 1 / 300) - 1) * 100
    ax_hist.hist(growth, orientation='horizontal', bins=50,
                 color=colors[4], alpha=0.3)

    ax_hist.axhline(0, label='Break Even', color='k', linestyle=':')
    ax_hist.axhline(growth_med, label='Median', color=colors[0])
    ax_hist.axhline(growth_avg, label='Mean', color=colors[3])
    ax_hist.set_ylabel('Compound Growth Rate (%)')
    ax_hist.set_xlabel('Frequency')
    ax_hist.legend()

    plt.tight_layout()
    plt.show()

n_traj = NietszcheDice()
plot_dice_game(n_traj)

n_traj = NietszcheDice(cash=0.6)
plot_dice_game(n_traj)


# Optimal tradeoff
cash_frac = np.linspace(0, 1, 101)[::-1]
N = 10 # Multiple runs to smooth out the values
vals5 = np.zeros((len(cash_frac), N))
vals50 = vals5.copy()
vals95 = vals5.copy()
for i in range(N):
  for j, f in enumerate(cash_frac):
    traj = NietszcheDice(f)
    perc5, _ = getQuantilePath(traj, 0.05)
    perc50, _ = getQuantilePath(traj, 0.5)
    perc95, _ = getQuantilePath(traj, 0.95)
    vals5[j, i] += perc5
    vals50[j, i] += perc50
    vals95[j, i] += perc95

vals5_smooth = vals5.mean(axis=1)
vals50_smooth = vals50.mean(axis=1)
vals95_smooth = vals95.mean(axis=1)

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(vals5_smooth, label=r'$5^{th}$ Percentile')
plt.plot(vals50_smooth, label=r'$50^{th}$ Percentile')
plt.plot(vals95_smooth, label=r'$95^{th}$ Percentile')
plt.scatter(vals5_smooth.argmax(), vals5_smooth.max(),
            marker='*', s=200)
plt.scatter(vals50_smooth.argmax(), vals50_smooth.max(),
            marker='*', s=200)
plt.scatter(vals95_smooth.argmax(), vals95_smooth.max(),
            marker='*', s=200)
plt.xlabel('Percentage of Wealth Wagered')
plt.ylabel('Ending Wealth')
plt.title('Optimal Bet Size')
plt.semilogy()
plt.legend()
plt.show()

def discreteKellyCriterion(x: float, returns: list, probs: list):
  return np.prod([(1 + b * x)**p for b, p in zip(returns, probs)]) - 1

probs = np.repeat(1/6, 6)
returns = [-0.5, 0.05, 0.05, 0.05, 0.05, 0.5]
g = np.array([discreteKellyCriterion(f, returns, probs)
  for f in cash_frac])
g *= 100

plt.figure(figsize=(12, 8))
plt.plot(cash_frac, g)
plt.xlabel('Fraction Bet')
plt.ylabel('Compound Growth Rate (%)')
plt.title('Optimal Bet Size According to the Kelly Criterion')
plt.show()

# Playing Dice with Insurance
f = 0.09
insurance = np.array([6, 0, 0, 0, 0, 0])
returns = np.array([0.5, 1.05, 1.05, 1.05, 1.05, 1.5])

ins_rets = f * insurance + (1 - f) * returns
print(f'Mean Returns with Insurance {(ins_rets.mean() - 1) * 100:.1f}%')

ins_gm = (np.power(np.prod(np.power(ins_rets, 50)), 1/300) - 1) * 100
print(f'Geometric Mean with Insurance {ins_gm:.1f}%')


def NietszcheDiceIns(ins_frac: float=0,
                     dice_returns: list=[0.5, 1.05, 1.05, 1.05, 1.05, 1.5],
                     ins_returns: list=[6, 0, 0, 0, 0, 0],
                     rolls: int=300, samples: int=10000):
  bet = 1 - ins_frac
  adj_returns = f * np.asarray(ins_returns) + bet * np.asarray(returns)
  roll_sims = np.random.choice(adj_returns,
    size=(rolls, samples)).reshape(-1, rolls)
  return roll_sims.cumprod(axis=1)

# With insurance
ins_frac = 0.09
ins_traj = NietszcheDiceIns(ins_frac)
plot_dice_game(ins_traj)