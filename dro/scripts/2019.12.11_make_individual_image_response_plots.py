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

from dro.modules import eye_tracking as et

def make_image_response_plot(osid):
    ed = et.open_eye_data(osid)
    bd = et.open_behavior_data(osid)
    pupil_responses,running_responses = et.get_individual_image_responses(ed,bd)
    fig,ax = et.plot_individual_image_responses(pupil_responses,running_responses)
    return fig,ax

if __name__ == '__main__':
    osid = sys.argv[1]
    fig,ax = make_image_response_plot(osid)
    vbp.save_figure(fig,'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/eye_tracking_figures/individual_image_responses/osid={}'.format(osid))