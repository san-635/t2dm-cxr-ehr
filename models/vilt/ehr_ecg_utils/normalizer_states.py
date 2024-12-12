from __future__ import absolute_import
from __future__ import print_function

from vilt.ehr_ecg_utils.utils import Discretizer, Normalizer
from vilt.dataset import EHR_ECG_CXR_Dataset
from vilt.config import ex

import os
import copy
import numpy as np

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    
    # config parameters
    data_dir = _config['data_root']
    image_size = _config["image_size"]
    max_text_len = _config["max_text_len"]-1       # -1 since [class] token is prepended later
    timestep = _config["timestep"]
    impute = _config["impute_strategy"]
    max_ecg_len = _config["max_ecg_len"]-1         # -1 since [class] token is prepended later
    train_transform_keys = _config["train_transform_keys"]

    # Initialise discretizer and normalizers
    discretizer = Discretizer(timestep=timestep, impute_strategy=impute, store_masks=True, start_time='zero', config_path='vilt/ehr_ecg_utils/discretizer_config.json')
    ehr_normalizer = Normalizer()
    ecg_normalizer = Normalizer()

    # Initialise training dataset
    reader = EHR_ECG_CXR_Dataset(
        discretizer,
        ehr_normalizer,
        ecg_normalizer,
        split="train",
        dataset_dir=data_dir,
        max_text_len=max_text_len,
        max_ecg_len=max_ecg_len,
        transform_keys=train_transform_keys,
        image_size=image_size
    )

    n_samples = len(reader)
    for sample in range(n_samples):
        ret = reader.read_by_ts_filename(sample)
        
        # Extract and normalise EHR data
        ts_data = ret["ts_rows"]
        non_ts_data = {"gender": ret["gender"], "age": ret["age"], "family_history": ret["family_history"]}
        ehr_data = discretizer.transform(ts_data, non_ts_data, max_text_len)[0]
        ehr_normalizer._feed_data(ehr_data)
        
        # Extract and normalise ECG data
        ecg_data = np.array(ret["ecg"])
        ecg_normalizer._feed_data(ecg_data)
        
        if sample % 1000 == 0:
            print('Processed {} / {} train dataset samples'.format(sample, n_samples), end='\r')

    # Save both normalizer states as .pkl files
    ehr_file_path = os.path.join(os.path.dirname(__file__), 'ehr_normalizer__{}_{}h_{}'.format(max_text_len, timestep, impute))
    ecg_file_path = os.path.join(os.path.dirname(__file__), 'ecg_normalizer')
    print('Saving ehr_normalizer state to {} and ecg_normalizer state to {}'.format(ehr_file_path, ecg_file_path))
    ehr_normalizer._save_params(ehr_file_path)
    ecg_normalizer._save_params(ecg_file_path)