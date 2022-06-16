"""
Sampling from HMM
-----------------

This script shows how to sample points from a Hidden Markov Model (HMM):
we use a 4-state model with specified mean and covariance.

The plot show the sequence of observations generated with the transitions
between them. We can see that, as specified by our transition matrix,
there are no transition between component 1 and 3.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hmmlearn import hmm
from scipy.stats import norm
from SpiritData.logger import log
from SpiritData.primitive import get_ts, exception_handler
from SpiritData.data_from_datastream import get_datastream_ts_data

from Themes.HiddenMarkovModel.hidden_markov_model_params import instrs

from DatabaseConnections.research_base import conn_research
from DatabaseConnections.research_cursor import conn_psycopg_research


def get_color_codes(palette, n):
    from pylab import cm
    import matplotlib
    colors_l = []
    cmap = cm.get_cmap(palette, n)  # matplotlib color palette name, n colors
    for i in range(cmap.N):
        rgb = cmap(i)[:3]  # will return rgba, we take only first 3 so we get rgb
        colors_l.append(matplotlib.colors.rgb2hex(rgb))

    return colors_l


def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


class MarkovChain:

    def __init__(self, code: dict,
                 n_states=2,
                 initialize_params=True,
                 kind='db',
                 from_ds=True,
                 resample='daily',
                 last_period_only=False,
                 min_samples=100,
                 coeff=100):

        self.ds_code = code['code']
        self.dtype = code['dtype']
        self.start = code['start']

        self.kind = kind
        self.from_ds = from_ds
        self.n_states = n_states
        self.resample = resample
        self.chain_all_probs = None
        self.coeff = coeff
        self.min_samples = min_samples
        self.last_period_only = last_period_only
        self.initialize_params = initialize_params

        self.ri, self.ret = self.get_ri()

    def get_labels(self):
        if self.n_states == 2:
            return {'1': 'bull', '2': 'bear'}
        elif self.n_states == 3:
            return {'1': 'bull', '2': 'normal', '3': 'bear'}

    def get_ri(self):

        if self.kind == 'montecarlo':
            # Markov chain to decide the regime
            n, ret = 1000, 0.001
            p11, p22 = 0.99, 0.99

            P = np.array([[p11, 1 - p11],  # negative transition prob
                          [1 - p22, p22]])  # positive transition prob

            np.random.seed(0)
            i = 0
            switch = [0]
            I = np.arange(len(P))
            for _ in range(n):
                i = np.random.choice(I, p=P[i])
                switch.append(i)

            # Return based on regime
            switch_rets = pd.Series(switch).map({1: ret, 0: -ret})
            ret = switch_rets + np.random.normal(0, pd.Series(switch).map({1: 0.01, 0: 0.01}), len(switch_rets))
            ret = ret * self.coeff

            return None, ret

        elif self.kind == 'db':
            if self.from_ds:
                ri = get_datastream_ts_data(self.ds_code, fields=self.dtype, start_date=self.start).dropna()
                ri.columns = ri.columns.droplevel(1)
                ri = ri.iloc[:, 0]
                ri.index = [pd.to_datetime(i) for i in ri.index]
            else:
                ri = get_ts([self.ds_code]).iloc[:, 0]

            if self.resample == 'weekly':
                ri = ri.resample('W-FRI').last()
            elif self.resample == 'monthly':
                ri = ri.resample('M').last()
            elif self.resample == 'daily':
                pass

            ret = ri.pct_change().dropna() * self.coeff

            return ri, ret

        else:
            raise KeyError('Specify valid kind!')

    def check_none(self, fnc, name):
        if self.chain_all_probs is None:
            self.get_chain_probs()
        return fnc(name=name)

    def get_metric(self, name='filtered'):
        means = self.chain_all_probs[self.chain_all_probs.quantity.str.contains(name)].set_index(
            ['ref_date', 'quantity']).loc[:, 'amount'].unstack()
        return means

    def get_state(self):
        state = self.chain_all_probs[self.chain_all_probs.quantity.str.contains('last_obs')].set_index(
            ['ref_date', 'quantity']).loc[:, 'amount'].unstack()
        return state

    def get_chain_probs(self):
        q = """select * from hidden_markov where ds_code = '%s' and n_states = %s and resample='%s'"""
        df = pd.read_sql(q % (self.ds_code, str(self.n_states), self.resample), conn_research)
        self.chain_all_probs = df

    def chart(self, last_years=None):

        st = None
        add_label = ''
        if last_years is not None:
            st = pd.to_datetime('today') - pd.to_timedelta(last_years * 365, unit='D')
            add_label = '_last_%s_years' % str(last_years)

        init_par_lbl = ''
        if self.initialize_params:
            init_par_lbl = '_init_params'

        flt_probs = self.check_none(self.get_metric, name='filtered').truncate(before=st)
        means = self.check_none(self.get_metric, name='mu').truncate(before=st)
        sigmas = self.check_none(self.get_metric, name='sigma').truncate(before=st)
        transition_probs = self.check_none(self.get_metric, name='_to_').truncate(before=st)
        last_obs = self.check_none(self.get_metric, name='last_obs').truncate(before=st)
        ts = self.ri.truncate(before=flt_probs.index[0]).truncate(before=st)

        fig = plt.figure(figsize=(12, 11))
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        plots_dict = [{'df': means, 'ax': ax1, 'title': 'Means'},
                      {'df': sigmas, 'ax': ax2, 'title': 'Sigmas'},
                      {'df': transition_probs, 'ax': ax3, 'title': 'Transition Probs'},
                      {'df': last_obs, 'ax': ax4, 'title': 'State at $t$'}]

        for plot_dict in plots_dict:
            tp = plot_dict['df']
            tp.index.name = None
            tp.columns = [replace_all(lbl, self.get_labels()) for lbl in tp.columns]
            tp.plot(cmap='brg', ax=plot_dict['ax'])
            plot_dict['ax'].set_title(plot_dict['title'])

        fig.suptitle('Markov Switching - %s' % self.ds_code, fontsize=16)
        plt.tight_layout()
        pth = os.path.join(os.path.dirname(__file__), 'charts')
        fig.savefig(os.path.join(pth, '%s_HMM_metrics_%s_states%s%s.png' % (self.ds_code,
                                                                            str(self.n_states),
                                                                            add_label,
                                                                            init_par_lbl)))

        plt.close()

        fig2 = plt.figure(figsize=(10, 10))
        colors_l = get_color_codes('brg', self.n_states)
        axes_l = []
        axes_twinx_l = []
        filtered_dicts = []
        for i in range(self.n_states):
            axes_l.append(fig2.add_subplot(self.n_states, 1, i + 1))
            axes_twinx_l.append(axes_l[-1].twinx())
            if i == 0:
                lbl = 'Bull'
            elif i == self.n_states - 1:
                lbl = 'Bear'
            else:
                lbl = 'Normal'

            filtered_dicts.append({'s': flt_probs.iloc[:, i],
                                   'ax': axes_l[-1],
                                   'twin': axes_twinx_l[-1],
                                   'color': colors_l[i],
                                   'title': lbl})

        for flt in filtered_dicts:
            flt['ax'].fill_between(ts.index, ts, 0, color='lightgray', alpha=1)
            flt['twin'].plot(flt['s'].index, flt['s'], color=flt['color'], alpha=1.0)
            prob = round(flt['s'].values[-1], 2)
            flt['twin'].set_yticks([prob])
            flt['twin'].set_yticklabels([prob], size=9, color=flt['color'])
            flt['ax'].set_title('%s - %s' % (self.ds_code, flt['title']))
            flt['twin'].set_ylim(0, 1)

        plt.tight_layout()
        fig2.savefig(os.path.join(pth, '%s_HMM_filtered_probs_%s_states%s%s.png' % (self.ds_code,
                                                                                    str(self.n_states),
                                                                                    add_label,
                                                                                    init_par_lbl)))
        plt.close()

        c = {0: 'b', 1: 'r', 2: 'g'}
        l = {0: 'bull', 1: 'normal', 2: 'bear'}
        for i in range(0, self.n_states):
            mu = means.iloc[-1, :].values[i]
            std = sigmas.iloc[-1, :].values[i]
            x_axis = np.arange(mu - 4 * std, mu + 4 * std, 0.0001)
            plt.plot(x_axis, norm.pdf(x_axis, mu, std), linewidth=0.5, color=c[i])
            plt.fill_between(x_axis, norm.pdf(x_axis, mu, std), 0, alpha=0.2, color=c[i])

        plt.legend([l[k] for k in l.keys()])
        plt.title('%s - Distributions' % self.ds_code)
        plt.savefig(os.path.join(pth, '%s_distributions.png' % self.ds_code))
        plt.close()

    @exception_handler
    def compute_chain(self):
        # Params
        X = self.ret
        params_dict = {}

        if self.last_period_only:
            self.min_samples = len(X.index) - 1

        for cnt, dt in enumerate(X.index[self.min_samples:]):
            if not cnt % 100:
                log.info('Markov Chain Computation - %s - %s - # States: %s - Code: %s' %
                         (str(cnt), dt.strftime('%Y-%m-%d'), str(self.n_states), self.ds_code))

            model = hmm.GaussianHMM(n_components=self.n_states,
                                    covariance_type="full",
                                    verbose=False,
                                    n_iter=10000,
                                    init_params='s')

            if self.initialize_params:
                if cnt != 0:
                    model.means_ = stored_means
                    model.covars_ = stored_covars
                    model.transmat_ = stored_transmat

            to_fit = np.column_stack([X.loc[:dt]])
            model.fit(to_fit)

            stored_means = model.means_
            stored_covars = model.covars_
            stored_transmat = model.transmat_

            means = pd.Series(stored_means.flatten()).div(self.coeff)
            means = means.sort_values(ascending=False)
            idx = means.index
            last_obs_prob = pd.DataFrame(model.predict_proba(to_fit)).loc[:, idx].iloc[-1]
            last_obs_prob.index = ['last_obs_state_prob_' + str(i + 1) for i in range(self.n_states)]
            vols = pd.Series(np.sqrt(stored_covars.flatten())).div(self.coeff).loc[idx]
            probs = pd.DataFrame(stored_transmat).loc[idx, idx]

            # rename indexes
            vols.index = ['sigma_' + str(i + 1) for i in range(self.n_states)]
            means.index = ['mu_' + str(i + 1) for i in range(self.n_states)]
            probs.index = [str(i + 1) for i in range(self.n_states)]
            probs.columns = [str(i + 1) for i in range(self.n_states)]
            flt_idx = ['filtered_prob_' + str(i + 1) for i in range(self.n_states)]
            filtered_probs = pd.Series(np.dot(np.array(probs.T), np.array(last_obs_prob)), index=flt_idx)

            probs = probs.stack()
            probs.index = probs.index.get_level_values(0) + '_to_' + probs.index.get_level_values(1)
            params_dict[dt] = pd.concat([means, vols, probs, last_obs_prob, filtered_probs])

        df = pd.DataFrame(params_dict).T.stack().reset_index()
        df.columns = ['ref_date', 'quantity', 'amount']
        df['ds_code'] = self.ds_code
        df['n_states'] = self.n_states
        df['resample'] = self.resample

        # delete from table
        cur = conn_psycopg_research.cursor()
        q = """DELETE FROM hidden_markov where ds_code = '%s' and n_states = %s and resample ='%s'"""
        if self.last_period_only:
            from primitive import ids_sql_strings
            q += " and ref_date in %s"
            cur.execute(q % (self.ds_code,
                             str(self.n_states),
                             self.resample,
                             ids_sql_strings([i.strftime('%Y-%m-%d') for i in [X.index[-1]]], is_string=True)))

        else:
            cur.execute(q % (self.ds_code, str(self.n_states), self.resample))

        conn_psycopg_research.commit()

        # send to db
        df.to_sql('hidden_markov', conn_research, index=False, if_exists='append')


if __name__ == '__main__':
    signals_dict = {}
    ri_dict = {}
    for instr in instrs:
        for n in [2]:
            for init_params in [True, False]:
                mc = MarkovChain(n_states=n,
                                 code=instr,
                                 from_ds=True,
                                 resample='weekly',
                                 last_period_only=True,
                                 initialize_params=init_params)

                mc.compute_chain()
                mc.chart(last_years=5)
                mc.chart(last_years=None)
