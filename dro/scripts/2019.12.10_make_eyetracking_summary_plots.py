from visual_behavior import utilities as vbu
import visual_behavior.plotting as vbp
from visual_behavior.utilities import EyeTrackingData
from visual_behavior import database as db
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns


import sys


def open_eye_data(osid):
    print('on osid = {}'.format(osid))
    try:
        try:
            return EyeTrackingData(int(osid),data_source='mongodb')
        except ValueError:
            return EyeTrackingData(int(osid),data_source='filesystem')
    except:
        return None

def has_dlc_file(osid):
    return 'EyeDlcOutputFile' in list(db.get_well_known_files(osid)['name'])
    
def get_raw_points_path(osid):
    wkf = db.get_well_known_files(osid).set_index('name')
    return os.path.join(wkf.loc['EyeDlcOutputFile']['storage_directory'],wkf.loc['EyeDlcOutputFile']['filename'])

def get_ellipse_fit_path(osid):
    wkf = db.get_well_known_files(osid).set_index('name')
    return os.path.join(wkf.loc['EyeTracking Ellipses']['storage_directory'],wkf.loc['EyeTracking Ellipses']['filename'])

def get_raw_points(osid):
    if has_dlc_file(osid):
        raw_points = pd.read_hdf(get_raw_points_path(osid))
    return raw_points

def plot_eye_tracking_summary(eye_data):
    osid = eye_data.ophys_session_id

    if has_dlc_file(osid):
        raw_points = get_raw_points(osid)
        pupil_columns = sorted([c for c in np.unique(np.array(raw_points.columns.get_level_values(1))) if c.startswith('pupil')])
        palette = sns.color_palette("hls", len(pupil_columns))

    fig = plt.figure(figsize=(30,35))
    ax_ts = vbp.placeAxesOnGrid(fig,yspan=(0,0.1))

    df = eye_data.ellipse_fits['pupil']
    ax_ts.plot(df['blink_corrected_area'])
    ax_ts.set_xlabel('frame #')
    ax_ts.set_ylabel('ellipse fit area (pixels^2)')
    ax_ts.set_title('pupil area fit')

    frames_yspan = (0.2,1)
    total_frames_yspan = frames_yspan[1]-frames_yspan[0]

    ax_frames = []
    for row in np.arange(10):
        for col in np.arange(10):
            ax_frames.append(vbp.placeAxesOnGrid(fig,xspan=(row*0.1,(row+1)*0.1),yspan=(col*(total_frames_yspan/10)+frames_yspan[0],(col+1)*(total_frames_yspan/10)+frames_yspan[0])))

    frames = np.random.choice(np.arange(len(eye_data.ellipse_fits['pupil'])),size=100)

    for ii,ax in enumerate(ax_frames):
        frame = frames[ii]
        ax.imshow(eye_data.get_annotated_frame(frame,linewidth=8))
        ax.axis('off')
        ax.text(5,5,'frame {}'.format(frame),va='top',fontsize=6,color='yellow')
        if has_dlc_file(osid):
            for c,pupil_column in enumerate(pupil_columns):
                dft = raw_points[raw_points.columns.get_level_values(0)[0]][pupil_column]
                ax.plot(dft.loc[frame]['x'],dft.loc[frame]['y'],marker='o',color=palette[c],alpha=0.25,markersize=2)

    fig.tight_layout()
    plt.subplots_adjust(hspace=0,wspace=0)

    if has_dlc_file(osid):
        dlc_date = pd.to_datetime(os.path.getmtime(get_raw_points_path(osid))*1e9).strftime('%Y-%m-%d')
    else:
        dlc_date = 'no intermediate file'
    
    ellipse_date = pd.to_datetime(os.path.getmtime(get_ellipse_fit_path(osid))*1e9).strftime('%Y-%m-%d')
        
    fig.suptitle('ophys_session_id: {}\nDLC fit file dated {}\nellipse fit file dated {}'.format(osid,dlc_date, ellipse_date))

    return fig

if __name__ == '__main__':
    osid = sys.argv[1]
    eye_data = open_eye_data(osid)
    fig = plot_eye_tracking_summary(eye_data)
    vbp.save_figure(fig,'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/eye_tracking_figures/summary_figs/osid={}'.format(osid))