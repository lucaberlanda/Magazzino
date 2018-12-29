import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')


class KellyPrincipleBaseClass:
    def __init__(self, events_p, wrong_signal):

        events_p = 1


        """
        :param events_p:
        :param wrong_signal:
            
        """

        print('ciao')


events_with_prob_dict = {'A': 0.3, 'B': 0.4, 'C': 0.2, 'D': 0.1}
wrong_signal_prob = 0.9

wrong_signal_prediction_prob_dict = {'A': wrong_signal_prob,
                                     'B': wrong_signal_prob,
                                     'C': wrong_signal_prob,
                                     'D': wrong_signal_prob}

KellyPrincipleBaseClass(events_with_prob_dict, wrong_signal_prediction_prob_dict)