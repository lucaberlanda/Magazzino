import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from primitive import rebase_at_x
from data_from_datastream import get_datastream_ts_data


class Boostrap:

    def __init__(self, code='S&PCOMP', periods=240, sample_n=1000, other='cash', w=0):

        self.periods = periods
        self.sample_n = sample_n
        self.other = other
        self.w = w

        s = get_datastream_ts_data([code], 'RI', start_date=pd.to_datetime('1975-01-01')).dropna()
        s.index = [pd.to_datetime(i) for i in s.index]  # todo vectorize

        """
        s.columns = ['price']
        s['returns'] = s['price'].pct_change()
        s['log_returns'] = np.log(1 + s['returns'])
        s['annual_returns'] = np.exp(s['log_returns'].rolling(12).sum()) - 1
        s['annual_returns_2'] = np.exp(s['log_returns'].rolling(12).sum()) - 1  # same of previous value
        """

        self.rets = s.resample('M').last().pct_change().dropna()
        self.combine()

        self.df = None
        self.geom_mean = None
        self.arith_mean = None

    def combine(self):

        """
        The mixed asset allocation is chosen as a
        consistent 5th percentile CAGR to establish
        across all the blended  portfolios because doing so
        jointly maximizes their 5th percentile.
        """

        if self.other == 'cash':
            self.rets[self.other] = 0

    def median_CI(self):
        median = []
        for i in np.random.choice(self.rets.values.flatten(), size=[20, self.periods, self.sample_n]):
            df = rebase_at_x((pd.DataFrame(i) + 1).cumprod(), at=1)
            median.append(df.iloc[-1, :].median())

    def get_optimal_allocation_by_quantile(self):

        alloc_frac = np.linspace(0, 0.20, 101)
        N = 10

        vals5 = np.zeros((len(alloc_frac), N))
        vals50 = vals5.copy()
        vals95 = vals5.copy()

        for i in range(N):
            for j, f in enumerate(alloc_frac):
                print(i, j, f)
                traj = self.perform_bootstrap(allocation=f)

                perc5, _ = self.get_quantile_path(traj, 0.05)
                perc50, _ = self.get_quantile_path(traj, 0.5)
                perc95, _ = self.get_quantile_path(traj, 0.95)

                vals5[j, i] += perc5
                vals50[j, i] += perc50
                vals95[j, i] += perc95

        # Average our sample medians to smooth out the plot
        smooth5 = vals5.mean(axis=1)
        smooth50 = vals50.mean(axis=1)
        smooth95 = vals95.mean(axis=1)

    def get_quantile_path(trajectories: np.array, q: float = 0.5):
        quantile = np.quantile(trajectories[:, -1], q=q)
        path = trajectories[np.abs(quantile - trajectories[:, -1]).argmin()]
        return quantile, path

    def perform_bootstrap_old(self):
        # np.random.seed(1234)
        df = pd.DataFrame(np.random.choice(self.rets.values.flatten(), size=[self.periods, self.sample_n]))
        df = rebase_at_x((df + 1).cumprod(), at=1)
        self.df = df.sort_values(df.index[-1], axis=1)
        self.geom_mean = self.df.apply(lambda x: stats.gmean(x.pct_change().dropna().values + 1) - 1)
        self.arith_mean = self.df.apply(lambda x: x.pct_change().mean())

    def perform_bootstrap(self,
                          allocation: float = 0):

        sims = np.random.choice(np.arange(len(self.rets.index)), size=(self.sample_n, self.periods))
        bootstrapped_rets_single = self.rets.values[sims]

        # Calculate returns
        bootstrapped_rets = np.matmul(bootstrapped_rets_single, [1 - allocation, allocation]).T
        bootstrapped_ris = rebase_at_x(pd.DataFrame(bootstrapped_rets + 1).cumprod(), at=1)
        return bootstrapped_ris

    def plot_means_distribution(self):
        self.geom_mean.plot(kind='hist', bins=100, alpha=0.5)
        self.arith_mean.plot(kind='hist', bins=100, alpha=0.5)
        plt.show()

    def plot(self):

        fig = plt.figure(figsize=(11, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=(3, 1))
        ax = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])

        path50 = self.df.iloc[:, int(self.sample_n / 2)]
        path_95 = self.df.iloc[:, int(self.sample_n / 100 * 95)]
        path_5 = self.df.iloc[:, int(self.sample_n / 100 * 5)]
        path_avg = self.df.mean(axis=1)

        self.df.plot(legend=False, logy=True, color='gray', alpha=0.1, ax=ax)
        path50.plot(color='black', ax=ax)
        path_avg.plot(color='blue', ax=ax)
        self.df.iloc[:, int(self.sample_n / 100 * 5)].plot(color='red', ax=ax, label='5th percentile')
        self.df.iloc[:, int(self.sample_n / 100 * 95)].plot(color='green', ax=ax, label='95th percentile')

        growth = (np.power(self.df.iloc[-1, :], 1 / self.periods) - 1) * 100
        growth_med = (np.power(path50.iloc[-1], 1 / self.periods) - 1) * 100
        growth_avg = (np.power(path_avg.iloc[-1], 1 / self.periods) - 1) * 100
        growth_95 = (np.power(path_95.iloc[-1], 1 / self.periods) - 1) * 100
        growth_5 = (np.power(path_5.iloc[-1], 1 / self.periods) - 1) * 100

        ax_hist.hist(growth, orientation='horizontal', bins=100, alpha=0.3, color='gray')

        # ax_hist.axhline(0, label='Break Even', color='k', linestyle=':')

        ax_hist.axhline(growth_med, label='Median', color='black')
        ax_hist.axhline(growth_avg, label='Mean', color='blue')
        ax_hist.axhline(growth_95, label='Mean', color='red')
        ax_hist.axhline(growth_5, label='Mean', color='green')
        ax_hist.set_ylabel('Compound Growth Rate (%)')
        ax_hist.set_xlabel('Frequency')
        ax_hist.legend()

        plt.tight_layout()
        plt.show()


btp = Boostrap()
btp.get_optimal_allocation_by_quantile()
btp.perform_bootstrap()
btp.plot()
