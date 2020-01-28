import matplotlib.pyplot as pyplot
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
            line = dict(color='black', width=3),
            name = 'grand average'
        )
    )

    return fig

def designate_flashes_plotly(fig,omit=None,pre_color='blue',post_color='blue',alpha=0.25,plotnumbers=[1],lims=[-10,10]):
    '''add vertical spans to designate stimulus flashes'''
    
    post_flashes = np.arange(0,lims[1],0.75)
    post_flash_colors = np.array([post_color]*len(post_flashes))
    pre_flashes = np.arange(-0.75,lims[0]-0.001,-0.75)
    pre_flash_colors = np.array([pre_color]*len(pre_flashes))

    flash_times = np.hstack((pre_flashes,post_flashes))
    flash_colors = np.hstack((pre_flash_colors,post_flash_colors))

    shape_list = list(fig.layout.shapes)
    
    for plotnumber in plotnumbers:
        for flash_start,flash_color in zip(flash_times,flash_colors):
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