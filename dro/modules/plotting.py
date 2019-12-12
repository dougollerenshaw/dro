import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import plotly.express as px


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

    return fig