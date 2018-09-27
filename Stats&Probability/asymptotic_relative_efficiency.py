import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ARC_list = []

for i in range(100):

    df = pd.DataFrame(np.random.randn(1000, 100))
    absolute_values = df.apply(lambda x: abs(x), axis=0)

    mean_deviation = absolute_values.mean()
    mean_deviation_exp_value_squared = np.square(mean_deviation.mean())
    mean_deviation_variance = mean_deviation.var()

    ARC_denominator = mean_deviation_variance / mean_deviation_exp_value_squared

    std_deviation = df.std()
    std_deviation_exp_value_squared = np.square(std_deviation.mean())
    std_deviation_variance = std_deviation.var()

    ARC_numerator = std_deviation_variance / std_deviation_exp_value_squared

    ARC = ARC_numerator / ARC_denominator
    ARC_list.append(ARC)

ARC_series = pd.Series(ARC_list)
ARC_series.plot()
plt.show()
