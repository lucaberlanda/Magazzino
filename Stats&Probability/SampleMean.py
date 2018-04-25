import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')


def why_divide_sample_variance_by_n_minus_one(min_sample_size, max_sample_size, trials=1000,
                                              plot_sample_st_dev_hist=False):
    expected_sample_stdev_dict = {}

    for i in range(min_sample_size, max_sample_size):  # iterate over the different (low) sample sizes

        print('Sample size:' + str(i) + ' (repeated ' + str(trials) + ' times)')

        simulation = pd.DataFrame(np.random.randn(i, trials))
        mean = simulation.mean()
        sample_std_dev_unbiased = simulation.std()  # Normalized by N-1 by default
        sample_std_dev_biased = simulation.std(ddof=0)  # Normalized by N-1 by default

        # sample_std_dev_biased = np.sqrt(simulation.apply(lambda x: ((x - x.mean())**2).sum() / i))
        mean_variance = mean.var()
        """MEAN VARIANCE: it is equal to sigma^2 / n (it will start at 0.5 with 2 sample size and 0.33 with 3 sample 
        size and so on...). The sample mean is an estimator of the population mean. It is strongly consistent 
        (therefore it is also weakly consistent) because its MSE tends to zero as the sample size gets larger."""

        sample_std_dev = pd.concat([sample_std_dev_unbiased, sample_std_dev_biased], axis=1)
        sample_std_dev.columns = ['Unbiased', 'Biased']

        if plot_sample_st_dev_hist:
            """ When the sample size is so low, the standard deviation is skewed to the right, i.e. we are more likely 
            to underestimate it; that's why the sample variance denominator should be n - 1, trying yo correct the 
            bias"""

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist(sample_std_dev.loc[:, 'Unbiased'])
            ax.hist(sample_std_dev.loc[:, 'Biased'])
            ax.set_xlabel('Estimated standard deviation')
            ax.set_ylabel('frequency')
            ax.set_title('ESTIMATED STANDARD DEVIATION HISTOGRAM (sample: ' + str(i) + ')')
            plt.show()
            plt.close()

        expected_sample_stdev_dict[i] = sample_std_dev.mean()

    expected_sample_stdev = pd.DataFrame(expected_sample_stdev_dict).T

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(expected_sample_stdev)
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Expected Standard Deviation')
    ax.set_title('ESTIMATED STANDARD DEVIATION')
    ax.legend(loc="upper right")
    plt.show()
    plt.close()

    simulation2 = pd.DataFrame(np.random.randn(1000))
    sample_std_dev_unbiased = simulation2.expanding(min_periods=3).std()
    sample_std_dev_biased = simulation2.expanding(min_periods=3).std(ddof=0)
    cc = pd.concat([sample_std_dev_unbiased, sample_std_dev_biased], axis=1)
    cc.columns = ['Unbiased', 'Biased']
    cc.plot()
    plt.show()


why_divide_sample_variance_by_n_minus_one(2, 50, plot_sample_st_dev_hist=False)
