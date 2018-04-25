import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')


class KellyPrincipleBaseClass:
    
    def __init__(self, p, bet_percentage, quote, sample=500):

        """
        :param p: float; probability of winning
        :param bet_amount: float; between 0 and 1 dollar; it correspond to the percentage of money bet
        :param sample: int; how many draws should be taken (T)

        """

        self.p = p
        self.quote = quote + 1
        self.bet_percentage = bet_percentage
        self.sample = sample
        self.draws = pd.Series(np.random.choice([0, self.quote], size=(self.sample), p=[1-self.p, self.p]))

    def final_value_and_growth(self):
        initial_capital = 1
        final_capital = (1 + self.bet_percentage) ** (self.draws.values.tolist().count(self.quote)) * \
                        (1 - self.bet_percentage) ** (self.draws.values.tolist().count(0)) * initial_capital

        growth = np.log2(final_capital)/self.sample
        return final_capital, growth

    def kelly_capital_evolution(self):
        capital = 1
        capital_evolution = {}
        for i, j in zip(self.draws, range(len(self.draws))):
            capital_evolution[j] = capital
            bet_capital = capital * self.bet_percentage
            remained_capital = capital - bet_capital
            capital_after_bet = (i * bet_capital) + remained_capital
            capital = capital_after_bet

        return pd.Series(capital_evolution)

expected_value_dict = {}
avg_growth_dict = {}
bet_amount_range = np.arange(0.05, 1, 0.1)
winning_prob = 0.8
quote = 1
theoretical_optimum = (winning_prob*(quote + 1) - 1) / quote
capital_evolution_dict = {}

for j in bet_amount_range:
    # j is the % of capital bet
    growth_dict = {}
    final_values_dict = {}
    sample = KellyPrincipleBaseClass(winning_prob, j, quote)
    capital_evolution = sample.kelly_capital_evolution()
    capital_evolution_dict[j] = capital_evolution
    final_value, growth = sample.final_value_and_growth()
    final_values_dict[j] = final_value
    growth_dict[j] = growth

    final_values = pd.Series(final_values_dict)
    growth_s = pd.Series(growth_dict)
    avg_growth_dict[j] = growth_s.mean()

pd.DataFrame(capital_evolution_dict).plot(logy=True, legend=True)
plt.show()

ax = plt.subplot(1,1,1)
avg_growth = pd.Series(avg_growth_dict)
avg_growth.index = bet_amount_range
print(avg_growth.index)
ax.bar(avg_growth.index, avg_growth.values.tolist(), width=0.025, align='center')
ax.set_xticks(avg_growth.index)
ax.set_xlabel('Bet Amount', fontsize=15)
ax.set_ylabel('Capital Growth', fontsize=15)

# place a text box in upper left in axes coords
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textstr = 'winning probability: %.2f\ntheoretical optimum:%.2f'%(winning_prob, theoretical_optimum)
ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', bbox=props)
plt.tight_layout()
plt.show()