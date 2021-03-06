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

from utilities import event_triggered_response, subtract_mean, convert_to_fraction, plot_event_triggered_response

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


def get_event_triggered_responses(ed,bd):
    responses = {}
    responses['ophys_session_id'] = ed.ophys_session_id
    responses['pupil_area'] = {}
    responses['running'] = {}
    
    datastream_map = {
        'pupil_area':{'input_data':ed.ellipse_fits['pupil'], 'parameter':'normalized_blink_corrected_area'},
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
    

def open_eye_data(osid):
    print('opening eye data for osid = {}'.format(osid))
    try:
        return EyeTrackingData(int(osid),data_source='filesystem')
    except:
        return None

def get_response_data(osid):
    t0 = time.time()
    ed = open_eye_data(osid)
    bd = open_behavior_data(osid)
    try:
        v = get_event_triggered_responses(ed,bd)
    except Exception as e:
        print('failed on {} with {}'.format(osid,e))
    print('done with osid = {}, that took {} seconds'.format(osid,time.time() - t0))
    return v

def open_behavior_data(osid):
    print('opening behavior data for osid = {}'.format(osid))
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


def get_individual_image_responses(ed,bd, relative_to='first_presentations', eye_parameter='blink_corrected_area'):
    '''
    parameters:
        ed = eye data
        bd = behavior data
        relative_to:
            'first_presentations' references to first presentation of every image (t0 = first_presentation of new)
            'last_presentations' references to last presentation of every image (t0 = first presentation of new)
    '''
    images = [im for im in np.sort(bd.visual_stimuli['image_name'].unique()) if im.startswith('im')]

    pupil_responses = {}
    running_responses = {}
    for image in images:
        if relative_to == 'first_presentations':
            first_times = bd.visual_stimuli.query('image_name == @image and consecutive_flashes == 0')['time']
        elif relative_to == 'last_presentations':
            first_times = bd.visual_stimuli.query('last_image_name == @image and consecutive_flashes == 0')['time']
        pupil_responses[image] = convert_to_fraction(event_triggered_response(ed.ellipse_fits['pupil'],parameter=eye_parameter,event_times=first_times,t_before=15,t_after=15))
        running_responses[image] = event_triggered_response(bd.running,parameter='speed',event_times=first_times,t_before=10,t_after=10)
    
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