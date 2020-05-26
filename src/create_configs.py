"""
IEOR 262B MATHEMATICAL PROGRAMMING II
SUBSET SELECTION PROJECT

Author: Miguel Fernandez Montes

Script to create experiment configuration CSV
"""


import os
import itertools
import numpy as np
import pandas as pd
import config


def main():
    # snr values
    snr_min = 0.05
    snr_max = 8
    snr_levels = np.geomspace(snr_min, snr_max, num=10).round(2)

    # beta types
    beta_types = [1, 2, 3, 5]

    # correlation factors
    corr_values = [0, 0.35, 0.7, 0.9]

    # data settings
    settings = ['low', 'mid', 'high']

    # experiments_path = '/home/miguel/Berkeley/SpringSemester/INDENG262B/Project/subset-selection/experiments'
    experiments_path = config.experiments_path
    filepath = os.path.join(experiments_path, 'experiment_settings.csv')

    combinations = itertools.product(settings, beta_types, corr_values, snr_levels)
    df = pd.DataFrame(data=combinations, columns=['setting', 'beta_type', 'corr', 'snr'])
    df.to_csv(filepath)


if __name__ == '__main__':
    main()
