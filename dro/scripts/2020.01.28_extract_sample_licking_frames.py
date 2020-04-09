from allensdk.brain_observatory.behavior.behavior_project_cache import BehaviorProjectCache as bpc
from visual_behavior.translator.allensdk_sessions import sdk_utils
import visual_behavior.utilities as vbu
import scipy.misc
import os

from visual_behavior import database as db
import pandas as pd
import numpy as np

from tqdm import tqdm

def save_frames(oeid,positive_example_path='/ssd1/lick_detection_model/positive_examples',negative_example_path='/ssd1/lick_detection_model/negative_examples'):
    manifest_path = "/home/dougo/manifest.json"
    cache = bpc.from_lims(manifest=manifest_path)
    
    osid = sdk_utils.get_osid_from_oeid(oeid,cache)
    
    well_known_files = db.get_well_known_files(osid).set_index('name')

    sync_path = ''.join(well_known_files.loc['OphysRigSync'][['storage_directory', 'filename']].tolist())
    sync_data = vbu.get_sync_data(sync_path)

    behavior_video_path = ''.join(well_known_files.loc['RawBehaviorTrackingVideo'][['storage_directory', 'filename']].tolist())
    behavior_video = vbu.Movie(behavior_video_path, sync_timestamps=sync_data['cam1_exposure_rising'])

    session = cache.get_session_data(oeid)
    session.licks['frame'] = session.licks['time'].map(lambda t:vbu.find_nearest_index(t,sync_data['cam1_exposure_rising']))

    lick_frames = session.licks['frame']
    non_lick_frames = list(set(np.arange(len(sync_data['cam1_exposure_rising']))) - set(lick_frames))

    for frames,savepath in zip([lick_frames,non_lick_frames],[positive_example_path,negative_example_path]):
        for frame in tqdm(np.random.choice(frames,100,replace=False)):
            image = behavior_video.get_frame(frame)
            filename = 'oeid_{}__frame_{}.png'.format(oeid,frame)
            scipy.misc.imsave(os.path.join(savepath, filename), image)

    
if __name__ == "__main__":
    oeid = 938001540
    save_frames(oeid)