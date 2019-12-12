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
        
        d.update({'{}'.format(ii):np.interp(d[time_key], t, y)})
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
    cols = [col for col in df.columns if col not in ['t','time']]
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

def get_session_name(input_name):
    session_name_dict = {
        'OPHYS_1_images_A':'A1', 
        'OPHYS_2_images_A_passive':'A2', 
        'OPHYS_3_images_A':'A3',
        'OPHYS_4_images_B':'B1', 
        'OPHYS_5_images_B_passive':'B2', 
        'OPHYS_6_images_B':'B3'
    }
    return session_name_dict[input_name]

def get_session_color(input_name):
    session_names = ['A1','A2','A3','B1','B2','B3']
    palette = get_colors_for_session_numbers()
    return palette[session_names.index(input_name)]

def get_colors_for_session_numbers():
    reds = sns.color_palette('Reds_r', 6)[:5][::2]
    blues = sns.color_palette('Blues_r', 6)[:5][::2]
    return reds+blues


def get_individual_image_responses(ed,bd):
    images = [im for im in np.sort(bd.visual_stimuli['image_name'].unique()) if im != 'grayscreen']

    pupil_responses = {}
    running_responses = {}
    for image in images:
        first_times = bd.visual_stimuli.query('image_name == @image and consecutive_flashes == 0')['time']
        pupil_responses[image] = event_triggered_response(ed.ellipse_fits['pupil'],parameter='blink_corrected_area',event_times=first_times,t_before=5,t_after=30)
        running_responses[image] = event_triggered_response(bd.running,parameter='speed',event_times=first_times,t_before=5,t_after=30)
    
    return pupil_responses,running_responses

def plot_individual_image_responses(pupil_responses,running_responses):
    palette = sns.color_palette("brg", 8)

    assert pupil_responses.keys() == running_responses.keys(), 'keys in response dictionaries must match'
    images = sorted(list(pupil_responses.keys()))

    fig,ax=plt.subplots(2,1,sharex=True,figsize=(12,7))
    for ii,image in enumerate(images):

        df = running_responses[image]
        cols = [c for c in df if c not in ['t','time']]
        ax[0].plot(
            df['time'],
            df[cols].mean(axis=1),
            color = palette[ii],
        )

        df = subtract_mean(pupil_responses[image])
        cols = [c for c in df if c not in ['t','time']]
        ax[1].plot(
            df['time'],
            df[cols].mean(axis=1),
            color = palette[ii],
        )

    ax[0].set_title('running speed')
    ax[0].set_ylabel('cm/s')
    ax[1].set_title('pupil diameter')
    ax[1].set_ylabel('change in pupil diameter (pixels**2)')
    ax[-1].set_xlabel('time from change (s)')

    ax[0].legend(images)
    for axis in ax:
        designate_flashes(axis,pre_color='darkgreen',post_color='blue')
    axis.set_xlim(-5,30)
    fig.tight_layout()
    sns.despine()
    
    return fig,ax