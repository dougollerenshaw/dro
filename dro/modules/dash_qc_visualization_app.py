#!/usr/bin/env python

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import numpy as np
import datetime
import pytz
import platform
from flask import send_file
import datetime

from allensdk.brain_observatory.behavior.behavior_project_cache import BehaviorProjectCache as bpc

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Visual Behavior Data QC'

# app = dash.Dash()
app.config['suppress_callback_exceptions'] = True

def load_data():
    manifest_path = "/home/dougo/manifest.json"
    cache = bpc.from_lims(manifest=manifest_path)

    table = cache.get_experiment_table().sort_values(by='date_of_acquisition',ascending=False).reset_index()

    table = cache.get_experiment_table().sort_values(by='date_of_acquisition',ascending=False).reset_index()
    container_ids = table['container_id'].unique()
    list_of_dicts = []
    for container_id in container_ids:
        subset = table.query('container_id == @container_id').sort_values(by='date_of_acquisition',ascending=True).drop_duplicates('ophys_session_id').reset_index()
        temp_dict = {
            'container_id':container_id,
            'container_workflow_state':table.query('container_id == @container_id')['container_workflow_state'].unique()[0],
            'first_acquistion_date':subset['date_of_acquisition'].min().split(' ')[0],
        }
        for idx,row in subset.iterrows():
            temp_dict .update({'session_{}'.format(idx):row['session_name']})
            

        list_of_dicts.append(temp_dict)

    return pd.DataFrame(list_of_dicts)


table = load_data()

app.layout = html.Div([
    html.H4('Visual Behavior Data'),
    html.H4('  '),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i.replace('_',' '), "id": i} for i in table.columns],
        data=table.to_dict('records'),
        row_selectable="single",
        selected_rows=[0],
        page_size=15,
        filter_action='native',
        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center',
        },
        style_table={'overflowX': 'scroll'},
    ),

    html.H4('  '),
], className='container')



if __name__ == '__main__':
    app.run_server(debug=True, port=5678, host='0.0.0.0')