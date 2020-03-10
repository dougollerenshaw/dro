import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def initialize_legend(ax, colors, linewidth=1, linestyle='-', marker=None, markersize=8, alpha=1):
    """ initializes a legend on an axis to ensure that first entries match desired line colors
    Parameters
    ----------
    ax : matplotlib axis
        the axis to apply the legend to
    colors : list
        marker colors for the legend items
    linewdith : int, optional
        width of lines (default 1)
    linestyle : str, optional
        matplotlib linestyle (default '-')
    marker : str, optional
        matplotlib marker style (default None)
    markersize : int, optional
        matplotlib marker size (default 8)
    alpha : float, optional
        matplotlib opacity, varying from 0 to 1 (default 1)
    """
    for color in colors:
        ax.plot(np.nan, np.nan, color=color, linewidth=linewidth,
                linestyle=linestyle, marker=marker, markersize=markersize, alpha=alpha)


def get_fig_ax(fig, ax_index):
    '''
    a flexible method for getting figure and axis handles

    inputs:
        fig: matplotlib figure object
        ax_index: index of desired axis

    if fig is None
        will create a new figure and axis
        returns fig, ax

    if fig exists, but no axes exist:
        will create a single subplot axis
        returns fig, ax

    if fig exists and has associated axes:
        returns fig, ax[ax_index]
    '''
    if fig is None:
        fig, ax = plt.subplots()

    if fig and len(fig.get_axes()) == 0:
        ax = fig.add_subplot(111)
    else:
        ax = fig.get_axes()[ax_index]

    return fig, ax


def event_triggered_raster(df, x_value, event_times, fig=None, ax_index=0, var_name='', value_name='', t_before=10, t_after=10, plot_type='matplotlib', marker='|', color='black'):
    if plot_type == 'plotly':
        assert False, 'not yet implemented'
    elif plot_type == 'matplotlib':
        fig, ax = get_fig_ax(fig, ax_index)

        for ii, event_time in enumerate(event_times):
            query_string = "{0} > (@event_time - @t_before) and {0} < (@event_time + @t_after)".format(
                x_value)
            events = df.query(query_string).values - event_time
            ax.plot(
                events,
                ii*np.ones_like(events),
                marker=marker,
                color=color,
                linestyle='none'
            )

        ax.set_xlabel(var_name)
        ax.set_ylabel(value_name)


def event_triggered_response_plot(df, x_value, y_values, fig=None, ax_index=0, var_name='', value_name='', plot_type='matplotlib', cmap='viridis'):
    if plot_type == 'plotly':
        return plotly_event_triggered_response_plot(df, x_value, y_values, var_name=var_name, value_name=value_name)
    elif plot_type == 'matplotlib':
        cmap = cm.get_cmap(cmap)
        fig, ax = get_fig_ax(fig, ax_index)

        for ii, col in enumerate(y_values):
            ax.plot(df[x_value], df[col], alpha=0.25,
                    color=cmap(ii/(len(y_values)-1)))
        ax.plot(df[x_value], df[y_values].mean(
            axis=1), color='black', linewidth=3)

        ax.set_xlabel(var_name)
        ax.set_ylabel(value_name)

        return fig


def plotly_event_triggered_response_plot(df, x_value, y_values, var_name='line', value_name='value'):

    df_melted = pd.melt(
        df,
        id_vars=[x_value],
        value_vars=y_values,
        var_name=var_name,
        value_name=value_name
    )

    fig = px.line(
        df_melted,
        x=x_value,
        y=value_name,
        color=var_name,
        hover_name=var_name,
        render_mode="svg"
    )

    fig.add_trace(
        go.Scatter(
            x=df[x_value],
            y=df[y_values].mean(axis=1),
            line=dict(color='black', width=3),
            name='grand average'
        )
    )

    return fig


def designate_flashes(ax, omit=None, pre_color='blue', post_color='blue'):
    '''add vertical spans to designate stimulus flashes'''
    lims = ax.get_xlim()
    for flash_start in np.arange(0, lims[1], 0.75):
        if flash_start != omit:
            ax.axvspan(flash_start, flash_start+0.25,
                       color=post_color, alpha=0.25, zorder=-np.inf)
    for flash_start in np.arange(-0.75, lims[0]-0.001, -0.75):
        if flash_start != omit:
            ax.axvspan(flash_start, flash_start+0.25,
                       color=pre_color, alpha=0.25, zorder=-np.inf)


def designate_flashes_plotly(fig, omit=None, pre_color='blue', post_color='blue', alpha=0.25, plotnumbers=[1], lims=[-10, 10]):
    '''add vertical spans to designate stimulus flashes'''

    post_flashes = np.arange(0, lims[1], 0.75)
    post_flash_colors = np.array([post_color]*len(post_flashes))
    pre_flashes = np.arange(-0.75, lims[0]-0.001, -0.75)
    pre_flash_colors = np.array([pre_color]*len(pre_flashes))

    flash_times = np.hstack((pre_flashes, post_flashes))
    flash_colors = np.hstack((pre_flash_colors, post_flash_colors))

    shape_list = list(fig.layout.shapes)

    for plotnumber in plotnumbers:
        for flash_start, flash_color in zip(flash_times, flash_colors):
            if flash_start != omit:
                shape_list.append(
                    go.layout.Shape(
                        type="rect",
                        x0=flash_start,
                        x1=flash_start+0.25,
                        y0=-100,
                        y1=100,
                        fillcolor=flash_color,
                        opacity=alpha,
                        layer="below",
                        line_width=0,
                        xref='x{}'.format(plotnumber),
                        yref='y{}'.format(plotnumber),
                    ),
                )

    fig.update_layout(shapes=shape_list)
