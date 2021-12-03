import numpy as np
import pandas as pd


def BrownianMotion(S0, mu, sigma, T, I):
    paths = np.zeros((T + 1, I), np.float64)
    paths[0] = S0

    for t in range(1, T + 1):
        rand = np.random.standard_normal(I)
        paths[t] = paths[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * 1 +
                                         sigma * np.sqrt(1) * rand)

    return pd.DataFrame(paths)
