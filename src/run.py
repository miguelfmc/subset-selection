"""
IEOR 262B MATHEMATICAL PROGRAMMING II
SUBSET SELECTION PROJECT

Author: Miguel Fernandez Montes
"""

import os
import sys
import config

# Append directory to Python path
# This should be set according to local setup
# sys.path.append('/home/miguel/Berkeley/SpringSemester/INDENG262B/Project/subset-selection/')
# subset_selection_path = config.subset_selection_path
# sys.path.append(subset_selection_path)

import pandas as pd
from experiment import run_experiment, run_experiment_real


MODE = config.MODE
experiments_path = config.experiments_path


def main():
    configs = pd.read_csv(os.path.join(experiments_path, 'experiment_settings.csv'))

    if MODE == 'low_mid':
        # run only low and mid dimensional settings
        configs = configs[configs['setting'].isin(['low', 'mid'])]
        for idx, row in configs.iterrows():
            setting, beta_type, corr, snr = row['setting'], row['beta_type'], row['corr'], row['snr']
            run_experiment(setting, beta_type, corr, snr)

    elif MODE == 'all':
        # run all settings
        for idx, row in configs.iterrows():
            setting, beta_type, corr, snr = row['setting'], row['beta_type'], row['corr'], row['snr']
            run_experiment(setting, beta_type, corr, snr)

    elif MODE == 'high':
        # run only high dimensional setting
        configs = configs[configs['setting'] == 'high']
        for idx, row in configs.iterrows():
            setting, beta_type, corr, snr = row['setting'], row['beta_type'], row['corr'], row['snr']
            run_experiment(setting, beta_type, corr, snr)

    elif MODE == 'high_test':
        # run a subset of high dimensional setting
        configs = configs[(configs['setting'] == 'high') & (configs['corr'] == 0.7) & (configs['beta_type'] == 2)]
        for idx, row in configs.iterrows():
            setting, beta_type, corr, snr = row['setting'], row['beta_type'], row['corr'], row['snr']
            run_experiment(setting, beta_type, corr, snr)

    elif MODE == 'prostate':
        run_experiment_real('prostate')

    elif MODE == 'lymphoma':
        run_experiment_real('lymphoma')

    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
