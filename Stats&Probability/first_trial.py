import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

a = 4
c = a ** 6
print(c)
quit()


binomial_random_returns = [1.02, 0.98]
random_returns = pd.DataFrame(np.random.choice(binomial_random_returns, size=(100, 1000)))
random_time_series = random_returns.cumprod()  # compute cumulative product
random_time_series = random_time_series / random_time_series.iloc[0, :]
final_price = random_time_series.iloc[-1, :]
final_price_average = final_price.mean()
print(final_price_average)

sns.distplot(final_price, 50)
plt.show()



# random_time_series = random_returns.apply(lambda x: x * x[-1] )

