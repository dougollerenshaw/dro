from visual_behavior import utilities as vbu
import visual_behavior.plotting as vbp
from visual_behavior.utilities import EyeTrackingData
from visual_behavior import database as db
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import seaborn as sns

class BehaviorData(object):
    def __init__(self, ophys_session_id):
        
        self.ophys_session_id = ophys_session_id
        
        self.foraging_id = db.get_value_from_table('id', ophys_session_id, 'ophys_sessions', 'foraging_id')
        self.ophys_session_id = ophys_session_id
        self.behavior_session_uuid = self.foraging_id
        well_known_files = db.get_well_known_files(self.ophys_session_id)
        
        self.sync_path = ''.join(well_known_files.query('name=="OphysRigSync"')[['storage_directory','filename']].iloc[0].tolist())
        
        mongo_db = db.Database('visual_behavior_data')
        
        self.sync_path = ''.join(well_known_files.query('name=="OphysRigSync"')[['storage_directory','filename']].iloc[0].tolist())
        self.sync_data = vbu.get_sync_data(self.sync_path)
        
        self.running = self.get_running_data(self.foraging_id, mongo_db)
        self.omitted_stim = self.get_omitted_stim_data(self.foraging_id, mongo_db)
        self.time = pd.DataFrame({'t':self.sync_data['stim_vsync_falling']})
        self.trials = self.get_trials(self.foraging_id, mongo_db)
        self.visual_stimuli = self.get_stimuli(self.foraging_id, mongo_db)
        
        mongo_db.close()
        
        self.running['time'] = self.time
        self.omitted_stim['time'] = self.time.loc[self.omitted_stim['frame']].values
        self.visual_stimuli['time'] = self.time.loc[self.visual_stimuli['frame']].values
        
        self.count_flashes()
        
    def get_running_data(self,behavior_session_uuid, mongo_db):
        return pd.DataFrame(mongo_db['behavior_data']['running'].find_one({'behavior_session_uuid':behavior_session_uuid})['data'])

    def get_omitted_stim_data(self,behavior_session_uuid, mongo_db):
        return pd.DataFrame(mongo_db['behavior_data']['omitted_stimuli'].find_one({'behavior_session_uuid':behavior_session_uuid})['data'])

    def get_trials(self,behavior_session_uuid, mongo_db):
        return pd.DataFrame(mongo_db['behavior_data']['trials'].find({'behavior_session_uuid':behavior_session_uuid}))
    
    def get_stimuli(self,behavior_session_uuid, mongo_db):
        return pd.DataFrame(mongo_db['behavior_data']['visual_stimuli'].find_one({'behavior_session_uuid':behavior_session_uuid})['data'])
    
    def count_flashes(self):
        self.visual_stimuli = self.visual_stimuli[~self.visual_stimuli['image_name'].str.contains('Movie')]
        self.visual_stimuli['last_image_name'] = self.visual_stimuli['image_name'].shift()
        self.visual_stimuli['change_image'] = self.visual_stimuli['image_name'] != self.visual_stimuli['last_image_name']
        consecutive_flashes = 0
        for idx,row in self.visual_stimuli.iterrows():

            if row['change_image']:
                consecutive_flashes = 0
            else:
                consecutive_flashes += 1
            self.visual_stimuli.at[idx,'consecutive_flashes'] = consecutive_flashes
        
        next_flash = 0
        for idx,row in self.visual_stimuli.iloc[::-1].iterrows():
            if row['consecutive_flashes'] == 0:
                next_flash = -1
                self.visual_stimuli.at[idx,'flashes_to_next_change'] = 0
            elif next_flash == -1:
                next_flash = row['consecutive_flashes']
                self.visual_stimuli.at[idx,'flashes_to_next_change'] = 1
            else:
                self.visual_stimuli.at[idx,'flashes_to_next_change'] = next_flash - row['consecutive_flashes'] + 1

def event_triggered_response(df, parameter, event_times, t_before=10, t_after=10, sampling_rate=60):
    '''
    build event triggered response around a given set of events
    required inputs:
      df: dataframe of input data
      parameter: column of input dataframe to extract around events
      event_times: times of events of interest
    optional inputs:
      t_before: time before each of event of interest
      t_after: time after each event of interest
      sampling_rate: desired sampling rate of output (input data will be interpolated)
    output:
      dataframe with one time column ('t') and one column of data for each event
    '''
    if 't' in df.columns:
        time_key = 't'
    else:
        time_key = 'time'
        
    d = {time_key:np.arange(-t_before,t_after, 1/sampling_rate)}
    for ii,event_time in enumerate(np.array(event_times)):

        df_local = df.query("{0} > (@event_time - @t_before) and {0} < (@event_time + @t_after)".format(time_key))
        t = df_local[time_key] - event_time
        y = df_local[parameter]
        
        d.update({'event_{}'.format(ii):np.interp(d[time_key], t, y)})
    return pd.DataFrame(d)

def get_event_triggered_responses(ed,bd):
    responses = {}
    responses['ophys_session_id'] = ed.ophys_session_id
    responses['pupil_area'] = {}
    responses['running'] = {}
    
    datastream_map = {
        'pupil_area':{'input_data':ed.ellipse_fits['pupil'], 'parameter':'blink_corrected_area'},
        'running':{'input_data': bd.running, 'parameter':'speed'}
    }
    
    events = [
        {
            'type':'hit',
            'event_times': bd.time.loc[bd.trials.query('trial_type == "go" and response_type == "HIT"')['change_frame']]
        },
        {
            'type':'cr',
            'event_times': bd.time.loc[bd.trials.query('trial_type == "catch" and response_type == "CR"')['change_frame']]
        },
        {
            'type':'omitted_stim',
            'event_times': bd.omitted_stim['time']
        },
        {
            'type':'prechange_flashes',
#             'event_times': bd.visual_stimuli[bd.visual_stimuli['flashes_to_next_change'].between(1,1, inclusive=True)]['time']
            'event_times': bd.visual_stimuli.query('flashes_to_next_change == 1')['time']
        },
    ]
    
    for event in events:
        for datastream in ['pupil_area','running']:
        
            responses[datastream][event['type']] = event_triggered_response(
                datastream_map[datastream]['input_data'],
                datastream_map[datastream]['parameter'],
                event['event_times']
            )
    return responses

def subtract_mean(df):
    cols = [col for col in df.columns if col is not 't']
    for col in cols:
        df[col] = df[col] - df[col].mean(axis=0)
    return df

def plot_event_triggered_response(df, ax=None, title='', mean_subtract = False, time_zero_subtract = False, fg_color='black',bg_color='gray', show_traces=False):
    df = df.copy()
    if 't' in df.columns:
        time_key = 't'
    else:
        time_key = 'time'
        
    if mean_subtract:
        df = subtract_mean(df)
        
    if time_zero_subtract:
        t = df[time_key]
        zero_index = t[np.isclose(t,0)].index[0]
        cols = [c for c in df.columns if c != time_key]
        for col in cols:
            df[col] = df[col] - df[col].loc[zero_index]
    
    if ax==None:
        fig,ax=plt.subplots()
    cols = [col for col in df.columns if col is not 't']
    if show_traces:
        for col in cols:
            ax.plot(df[time_key], df[col],alpha=0.5,color=bg_color)
    ax.plot(df[time_key], df[cols].mean(axis=1), color=fg_color, linewidth = 3)
    ax.fill_between(
        df[time_key], 
        df[cols].mean(axis=1) + df[cols].std(axis=1), 
        df[cols].mean(axis=1) - df[cols].std(axis=1), 
        color=bg_color, 
        alpha=0.35
    )
    if ax==None:
        return fig,ax
    ax.set_title(title)
    
def designate_flashes(ax,omit=None,pre_color='blue',post_color='blue'):
    '''add vertical spans to designate stimulus flashes'''
    lims = ax.get_xlim()
    for flash_start in np.arange(0,lims[1],0.75):
        if flash_start != omit:
            ax.axvspan(flash_start,flash_start+0.25,color=post_color,alpha=0.25,zorder=-np.inf)
    for flash_start in np.arange(-0.75,lims[0]-0.001,-0.75):
        if flash_start != omit:
            ax.axvspan(flash_start,flash_start+0.25,color=pre_color,alpha=0.25,zorder=-np.inf)

def open_eye_data(osid):
    print('on osid = {}'.format(osid))
    try:
        try:
            return EyeTrackingData(int(osid),data_source='mongodb')
        except ValueError:
            return EyeTrackingData(int(osid),data_source='filesystem')
    except:
        return None

def get_response_data(osid):
    ed = open_eye_data(osid)
    bd = open_behavior_data(osid)
    v = get_event_triggered_responses(ed,bd)
    return v

def open_behavior_data(osid):
    print('on osid = {}'.format(osid))
    try:
        return BehaviorData(int(osid))
    except (AssertionError, OSError, TypeError):
        print('failed on {}'.format(osid))
        return None    

def test_func(v):
    print('input was {}'.format(v))