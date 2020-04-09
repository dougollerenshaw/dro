from visual_behavior import utilities as vbu
import visual_behavior.plotting as vbp
from visual_behavior import database as db
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import seaborn as sns


def event_triggered_response(df, parameter, event_times, time_key=None, t_before=10, t_after=10, sampling_rate=60):
    '''
    build event triggered response around a given set of events
    required inputs:
      df: dataframe of input data
      parameter: column of input dataframe to extract around events
      event_times: times of events of interest
    optional inputs:
      time_key: key to use for time (if None (default), will search for either 't' or 'time'. if 'index', use indices)
      t_before: time before each of event of interest
      t_after: time after each event of interest
      sampling_rate: desired sampling rate of output (input data will be interpolated)
    output:
      dataframe with one time column ('t') and one column of data for each event
    '''
    if time_key is None:
        if 't' in df.columns:
            time_key = 't'
        else:
            time_key = 'time'

    _d = {'time': np.arange(-t_before, t_after, 1/sampling_rate)}
    for ii, event_time in enumerate(np.array(event_times)):

        if time_key == 'index':
            df_local = df.loc[(event_time - t_before):(event_time + t_after)]
            t = df_local.index.values - event_time
        else:
            df_local = df.query(
                "{0} > (@event_time - @t_before) and {0} < (@event_time + @t_after)".format(time_key))
            t = df_local[time_key] - event_time
        y = df_local[parameter]

        _d.update({'event_{}_t={}'.format(ii, event_time): np.interp(_d['time'], t, y)})
    return pd.DataFrame(_d)


def event_triggered_raster(df, parameter, event_times, t_before=10, t_after=10, ):
    '''
    find discrete events surrounding list of event times
    (e.g.: licks or spike times relative to external events)
    required inputs:
      df: dataframe of input data
      parameter: column of input dataframe to extract around events
      event_times: times of events of interest

    optional inputs:
      t_before: time before each of event of interest
      t_after: time after each event of interest

    output: 
      dataframe with one column for each event. Each entry contains a list of events.
    '''
    _d = {}
    for ii, event_time in enumerate(np.array(event_times)):
        events = df.query(
            "{0} > (@event_time - @t_before) and {0} < (@event_time + @t_after)".format(parameter)).values
        _d.update({'event_{}_t={}'.format(ii, event_time): list(events - event_time}))
    return pd.DataFrame(_d)


def subtract_mean(df_in):
    df = df_in.copy()
    cols = [col for col in df.columns if col not in ['t', 'time']]
    for col in cols:
        df[col] = df[col] - df[col].mean(axis=0)
    return df


def convert_to_fraction(df_in):
    df = df_in.copy()
    cols = [col for col in df.columns if col not in ['t', 'time']]
    for col in cols:
        s = df[col]
        s0 = df[col].mean(axis=0)
        df[col] = (s - s0)/s0
    return df


def plot_event_triggered_response(df, ax=None, title='', mean_subtract=False, fraction=False, time_zero_subtract=False, fg_color='black', bg_color='gray', show_traces=False):
    df = df.copy()
    if 't' in df.columns:
        time_key = 't'
    else:
        time_key = 'time'

    if mean_subtract:
        df = subtract_mean(df)

    if fraction:
        df = convert_to_fraction(df)

    if time_zero_subtract:
        t = df[time_key]
        zero_index = t[np.isclose(t, 0)].index[0]
        cols = [c for c in df.columns if c != time_key]
        for col in cols:
            df[col] = df[col] - df[col].loc[zero_index]

    if ax == None:
        fig, ax = plt.subplots()
    cols = [col for col in df.columns if col != time_key]
    if show_traces:
        for col in cols:
            ax.plot(df[time_key], df[col], alpha=0.5, color=bg_color)
    ax.plot(df[time_key], df[cols].mean(axis=1), color=fg_color, linewidth=3)
    ax.fill_between(
        df[time_key],
        df[cols].mean(axis=1) + df[cols].std(axis=1),
        df[cols].mean(axis=1) - df[cols].std(axis=1),
        color=bg_color,
        alpha=0.35
    )
    if ax == None:
        return fig, ax
    ax.set_title(title)
