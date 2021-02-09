from visual_behavior.data_access import loading
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
import matplotlib.pyplot as plt
import visual_behavior.database as db
import pandas as pd
import numpy as np
import os
from visual_behavior.translator.foraging2 import data_to_change_detection_core
import visual_behavior.encoder_processing.running_data_smoothing as running_data_smoothing
import argparse

def open_and_plot_running(ophys_session_id, savedir='/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/session_plots/running_smoothing_check'):
    # get the experiment IDs for this session
    ophys_experiment_ids = db.lims_query('select id from ophys_experiments where ophys_session_id = {}'.format(ophys_session_id))
    
    # check to see whether result is single int or dataframe of ints
    if isinstance(ophys_experiment_ids, (int, np.int64)):
        # if single experiment for this session, use experiment
        ophys_experiment_id = ophys_experiment_ids
    else:
        # if many experiments for this session, use the first
        ophys_experiment_ids = ophys_experiment_ids['id'].tolist()
        ophys_experiment_id = ophys_experiment_ids[0]
        
    session = BehaviorOphysSession.from_lims(ophys_experiment_id)
    session_running_df_ORIGINAL = session.api.get_running_data_df(lowpass=True, zscore_threshold=np.inf)

    # get the running data directly from the PKL file to compare the VBA implementation of running smoothing
    osid = db.lims_query('select ophys_session_id from ophys_experiments where id = {}'.format(ophys_experiment_id))
    pkl_path = db.get_pkl_path(osid, 'ophys_session_id')
    data= pd.read_pickle(pkl_path)
    core_data = data_to_change_detection_core(data)

    # set the 'timestamps' column in core_data['running'] to be the SDK timestamps
    core_data['running']['timestamps'] = session_running_df_ORIGINAL.index[:len(core_data['running'])]

    # run the PKL running data through the VBA smoothing method
    processed_running_df = running_data_smoothing.process_encoder_data(
        core_data['running'], 
        time_column='timestamps', 
        v_max='v_sig_max', 
        filter_cutoff_frequency=4, 
        remove_outliers_at_wraps=True, 
        zscore_thresold=5
    )

    # make a plot

    t_mid = session_running_df_ORIGINAL['speed'].idxmax()
    t0 = t_mid - 15
    t1 = t_mid + 15
    fig, ax = plt.subplots(4,1,figsize=(15,12), sharex=True)
    ax[0].plot(
        processed_running_df.query('timestamps >= @t0 and timestamps <= @t1')['timestamps'],
        processed_running_df.query('timestamps >= @t0 and timestamps <= @t1')['v_sig'],
    )
    ax[0].set_title('v_sig')

    ax[1].plot(
        processed_running_df.query('timestamps >= @t0 and timestamps <= @t1')['timestamps'],
        processed_running_df.query('timestamps >= @t0 and timestamps <= @t1')['v_sig_unwrapped'],
    )
    ax[1].set_title('v_sig_unwrapped')

    ax[2].plot(
        processed_running_df.query('timestamps >= @t0 and timestamps <= @t1')['timestamps'],
        processed_running_df.query('timestamps >= @t0 and timestamps <= @t1')['speed_raw_pre_wrap_correction'],
    )
    ax[2].set_title('speed before artifact removal')

    legend_text = ['vba speed']
    line_pattern = ['-','--',':','-.']
    ax[3].plot(
        processed_running_df.query('timestamps >= @t0 and timestamps <= @t1')['timestamps'],
        processed_running_df.query('timestamps >= @t0 and timestamps <= @t1')['speed'],
        color='black',
        linewidth=6
    )
    for ii, z_threshold in enumerate([5,10,20,50]):
        session_running_df = session.api.get_running_data_df(lowpass=True, zscore_threshold=z_threshold)
        ax[3].plot(
            session_running_df.query('timestamps >= @t0 and timestamps <= @t1')['speed'],
            linewidth = 5-ii,
            linestyle = line_pattern[ii%len(line_pattern)]
        )
        legend_text.append('SDK speed, z threshold = {}'.format(z_threshold))
    ax[3].set_title('speed after artifact removal (SDK and VBA)')
    ax[3].legend(legend_text, loc='upper right')

    ax[3].set_xlabel('time (s)')
    ax[3].set_ylabel('speed (cm/s)')
    
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle('session_id = {}\nexperiment_id(s) = {}'.format(ophys_session_id, ophys_experiment_ids))
    
    fig.savefig(os.path.join(savedir, 'session_{}.png'.format(ophys_session_id)))
    
    return fig, ax

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run dash visualization app for VB production data')
    parser.add_argument(
        '--osid',
        type=int,
        default='',
        metavar='osid'
    )
    parser.add_argument(
        '--savedir',
        type=str,
        default='',
        metavar='save directory'
    )
    args = parser.parse_args()
    if args.savedir == '':
        savedir = None
    else:
        savedir = args.savedir
    open_and_plot_running(args.osid, savedir)
