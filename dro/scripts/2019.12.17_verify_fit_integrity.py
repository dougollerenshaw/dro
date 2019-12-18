import pandas as pd
import numpy as np
import sys


def get_pupil_and_eye_ratios(raw_point_path, ellipse_fit_path):
    '''
    for both the eye and the pupil fit
    gets ratio of:
        distance between left and right most fit points
        _______________________________________________
        width of ellipse fit

    this ratio should be close to 1

    '''
    def get_range(columns, frame=1000):
        '''
        get min and max x values
        '''
        min_x = np.inf
        max_x = -np.inf

        for column in columns:
            x = raw_points[raw_points.columns.get_level_values(
                0)[0]][column].loc[frame]['x']
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x

        return min_x, max_x

    # open raw points
    raw_points = pd.read_hdf(raw_points_path)

    # open ellipse fits
    ellipse_fits = {}
    for dataset in ['pupil', 'eye', 'cr']:
        ellipse_fits[dataset] = pd.read_hdf(ellipse_fit_path, dataset)

    # get pupil and eye columns from raw points
    pupil_columns = sorted([c for c in np.unique(
        np.array(raw_points.columns.get_level_values(1))) if c.startswith('pupil')])
    eye_columns = sorted([c for c in np.unique(
        np.array(raw_points.columns.get_level_values(1))) if c.startswith('eye')])

    # get ranges, then ratios
    min_x, max_x = get_range(pupil_columns)
    pupil_fit_to_point_ratio = (max_x - min_x) / (ellipse_fits['pupil'].loc[1000]['width']*2)

    min_x, max_x = get_range(eye_columns)
    eye_fit_to_point_ratio = (max_x - min_x) / (ellipse_fits['eye'].loc[1000]['width']*2)

    return pupil_fit_to_point_ratio, eye_fit_to_point_ratio


if __name__ == '__main__':
    raw_points_path = sys.argv[1]
    ellipse_fit_path = sys.argv[2]

    pr, er = get_pupil_and_eye_ratios(raw_points_path, ellipse_fit_path)
    print('\noutput:\n=========================================================================')
    print('pupil tracking point width to ellipse width ratio (should be ~1): {}'.format(pr))
    print('eye tracking point width to ellipse width ratio (should be ~1): {}'.format(er))
