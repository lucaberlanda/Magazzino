import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')


class KellyCriterion:

    def __init__(self, p, b=1, a=1):

        """
        :param p: float; probability of winning
        :param b: units won per unit of wagered amount (quote minus 1)
        :param a: units won per unit of wagered amount

        """
        self.p = p
        self.b = b
        self.a = a
        self.optimal_f = self.get_optimal_f()

    def get_optimal_f(self):
        q = 1 - self.p
        opt_f = (self.b * self.p - self.a * q) / (self.a * self.b)
        return opt_f

    def W_as_f_changes(self, steps=10000, f_step=0.025, graph=True):

        lgnd = True
        np.random.seed(5)
        draws = pd.Series(np.random.choice([self.b + 1, 1 - self.a], size=(steps), p=[self.p, 1 - self.p]))
        capital_evolution_dict = {}
        for f in np.arange(0.05, 0.95, f_step):
            capital_evolution_dict[round(f, 3)] = self.W_evolution(f, draws)

        W = pd.DataFrame(capital_evolution_dict)

        if graph:
            fig = plt.figure(1, figsize=(14, 6))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            if len(W.index) > 10:
                lgnd = False
            W.plot(logy=True, cmap='viridis', ax=ax1, legend=lgnd, alpha=0.6, lw=0.5)
            W.iloc[-1, :].plot(cmap='viridis', ax=ax2, logy=True)
            ax2.axvline(W.iloc[-1, :].idxmax(), color='r', linestyle='--', lw=1)
            ax2.axvline(self.optimal_f, color='b', linestyle='--', lw=1)
            ax1.set_title('Wealth Evolution')
            ax2.set_title('Wealth at time T - Theoretical Optimal: ' + str(round(self.optimal_f, 2)))

        return W, fig

    @staticmethod
    def W_evolution(bet_percentage, draws):
        capital = 1
        capital_evolution = {}
        for i, j in zip(draws, range(len(draws))):
            capital_evolution[j] = capital
            bet_capital = capital * bet_percentage
            remained_capital = capital - bet_capital
            capital_after_bet = (i * bet_capital) + remained_capital
            capital = capital_after_bet

        return pd.Series(capital_evolution)