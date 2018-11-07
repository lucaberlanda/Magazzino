import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""

Replication of: Taleb, Silent Risk 
[Mean Deviation vs Standard Deviation, Class Lecture Derivations]

Asymptotic Relative Efficiency (ARE)
ARE = with lim(N → inf) of (V(Std)/(E(Std)^2)) / (V(Mad)/E(Mad)^2)

"""

ARC_list = []
time_series_length = 10000
time_series_number = 1000
df = pd.DataFrame(np.random.randn(time_series_length, time_series_number))

for i in range(10, time_series_number):

    flt_df = df.iloc[:, :i]  # at each iteration, increase the number of timeseries by one
    absolute_values = flt_df.apply(lambda x: abs(x), axis=0)

    mean_deviation = absolute_values.mean()
    mean_deviation_exp_value_squared = np.square(mean_deviation.mean())
    mean_deviation_variance = mean_deviation.var()

    ARC_denominator = mean_deviation_variance / mean_deviation_exp_value_squared

    std_deviation = flt_df.std()
    std_deviation_exp_value_squared = np.square(std_deviation.mean())
    std_deviation_variance = std_deviation.var()

    ARC_numerator = std_deviation_variance / std_deviation_exp_value_squared

    ARC = ARC_numerator / ARC_denominator
    ARC_list.append(ARC)

ARC_series = pd.Series(ARC_list)
ARC_series.plot()
plt.show()
