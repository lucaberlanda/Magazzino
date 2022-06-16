import numpy as np
import pandas as pd

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import Viz.aestethics as aes

from primitive import rebase_at_x


def generate_ax(title=None, size=(8, 5), x_label=None, y_label=None):
    fig, ax = plt.subplots(facecolor=aes.back_c, figsize=size)
    ax.set_facecolor(aes.back_c)

    ax.spines['bottom'].set_color(aes.axis_c)
    ax.spines['top'].set_color(aes.back_c)
    ax.spines['right'].set_color(aes.back_c)
    ax.spines['left'].set_color(aes.axis_c)

    ax.xaxis.label.set_color(aes.axis_c)
    ax.yaxis.label.set_color(aes.axis_c)

    ax.tick_params(axis='x', colors=aes.axis_c)
    ax.tick_params(axis='y', colors=aes.axis_c)

    ax.set_ylabel(y_label, size=12)
    ax.set_xlabel(x_label, size=12)

    ax.set_title(title, size=18)

    return ax


def generate_axes(rows, columns):
    fig, ax = plt.subplots(facecolor=aes.back_c, figsize=(12, 7), nrows=rows, ncols=columns)
    for single_ax_v in ax:
        single_ax_v = [single_ax_v] if not isinstance(single_ax_v, list) else single_ax_v
        for single_ax in single_ax_v:
            single_ax.set_facecolor(aes.back_c)

            single_ax.spines['bottom'].set_color(aes.axis_c)
            single_ax.spines['top'].set_color(aes.axis_c)
            single_ax.spines['right'].set_color(aes.axis_c)
            single_ax.spines['left'].set_color(aes.axis_c)

            single_ax.xaxis.label.set_color(aes.axis_c)
            single_ax.tick_params(axis='x', colors=aes.axis_c)
            single_ax.yaxis.label.set_color(aes.axis_c)
            single_ax.tick_params(axis='y', colors=aes.axis_c)
    return ax


def plot_index(idx, color=None, title=None):
    ax = generate_ax()

    if color is None:
        color = 'blue'

    idx.plot(figsize=(8, 5), color=color, ax=ax, linewidth=1.5)

    if title is None:
        ax.set_title(idx.name, fontsize=aes.title_s)
    else:
        ax.set_title(idx.name, fontsize=aes.title_s)

    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_area(idx, color=None, title=None):
    ax = generate_ax()

    if color is None:
        color = 'blue'

    idx.plot(color=color, ax=ax, linewidth=0.1)
    ax.fill_between(idx.index, idx, facecolor='blue')

    ax.set_title(title, fontsize=aes.title_s)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def compare_indexes(strat_idx, ri_index, title='Real vs Rebuilt Index'):
    ax = generate_ax()

    strats = rebase_at_x(pd.concat([strat_idx, ri_index], axis=1).dropna())
    strats.plot(figsize=(8, 5), color=['#17A589', 'blue'], ax=ax, linewidth=1.5)
    ax.set_title(title, fontsize=aes.title_s)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_ts(ris, title=None, entry_point=None, **kwargs):
    ax = generate_ax()
    ris.plot(ax=ax, **kwargs)

    if entry_point is not None:
        df_scatter = ris.reset_index().loc[:, ['ref_date', entry_point['for']]]
        df_scatter.columns = ['dt', 'value']
        entry_point_values = df_scatter[df_scatter.dt == entry_point['at']]
        entry_point_values.plot(kind='scatter', x='dt', y='value', color='red', ax=ax, zorder=1000, s=50)

    ax.set_title(title, fontsize=aes.title_s)
    ax.legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_one_vs_many(one, many, title=None, entry_point=None):
    ax = generate_ax()
    one.plot(color='blue', ax=ax, linewidth=1.5, zorder=1000)
    many.plot(color='gray', ax=ax, linewidth=0.5, legend=False)

    if entry_point is not None:
        df_scatter = one.reset_index()
        df_scatter.columns = ['dt', 'value']
        entry_point_values = df_scatter[df_scatter.dt == entry_point]
        entry_point_values.plot(kind='scatter', x='dt', y='value', color='red', ax=ax, zorder=1001, s=30)

    ax.set_title(title, fontsize=12)
    ax.xaxis.set_label_text("")
    plt.tight_layout()
    plt.show()


def plot_two_vs_many(one, two, many, title=None, entry_point=None):
    ax = generate_ax()
    one.plot(color='blue', ax=ax, linewidth=1.5, zorder=1000)
    two.plot(color='black', ax=ax, linewidth=1.5, zorder=1000)
    many.plot(color='gray', ax=ax, linewidth=0.5, legend=False)

    if entry_point is not None:
        df_scatter = one.reset_index()
        df_scatter.columns = ['dt', 'value']
        entry_point_values = df_scatter[df_scatter.dt == entry_point]
        entry_point_values.plot(kind='scatter', x='dt', y='value', color='red', ax=ax, zorder=1001, s=30)

    ax.set_title(title, fontsize=12)
    ax.xaxis.set_label_text("")
    plt.tight_layout()
    plt.show()
