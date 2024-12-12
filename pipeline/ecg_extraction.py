import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import wfdb

# ECG SIGNAL EXTRACTION #

def filter_metadata(ecg_metadata_path, all_stays_path):
    ecg_metadata = pd.read_csv(ecg_metadata_path)
    ecg_metadata = ecg_metadata[['subject_id', 'ecg_time', 'path']]
    ecg_metadata.loc[:, 'ecg_time'] = pd.to_datetime(ecg_metadata['ecg_time'], format='%Y-%m-%d %H:%M:%S')
    print(f'{ecg_metadata.shape[0]} ECG studies exist for {len(set(ecg_metadata["subject_id"]))} subjects in MIMIC-IV-ECG v1.0.')

    all_stays = pd.read_csv(all_stays_path)
    all_stays = all_stays[['subject_id','admittime','dischtime']]
    all_stays.loc[:, 'admittime'] = pd.to_datetime(all_stays['admittime'], format='%Y-%m-%d %H:%M:%S')
    all_stays.loc[:, 'dischtime'] = pd.to_datetime(all_stays['dischtime'], format='%Y-%m-%d %H:%M:%S')
    all_stays = all_stays.groupby(['subject_id']).agg({'admittime': 'min', 'dischtime': 'max'}).reset_index()
    
    filtered_metadata = ecg_metadata.merge(all_stays, on='subject_id', how='inner')
    filtered_metadata = filtered_metadata[(filtered_metadata['ecg_time'] > filtered_metadata['admittime']) & (filtered_metadata['ecg_time'] < filtered_metadata['dischtime'])]
    return filtered_metadata[['subject_id','ecg_time','path']]

def extract_ecg(mimic_ecg_path, filtered_metadata, output_path):
    groups = filtered_metadata.groupby('subject_id')
    for subject_id, group in tqdm(groups, desc='Extracting ECG signals'):
        for _, row in group.sort_values('ecg_time').iterrows():
            ecg_study_id = row['path'].split('/')[-1]
            ecg_study_dir = mimic_ecg_path + row['path'][0:-9]
            signals, _ = wfdb.rdsamp(ecg_study_id, pn_dir=ecg_study_dir)
            if np.any(np.isnan(signals)):
                continue                                            # skip this ECG signal
            output_dir = os.path.join(output_path, str(subject_id))
            os.makedirs(output_dir, exist_ok=True)
            pd.DataFrame(signals).to_csv(os.path.join(output_dir, f'{ecg_study_id}o.csv'), index=False, header=False)
            break                                                   # keep only the chronologically first valid ECG signal