import os
import pandas as pd
from tqdm import tqdm
import shutil

# CXR IMAGE PREPROCESSING #

def read_metadata(metadata, input_path):
    metadata = metadata[metadata['ViewPosition'] == 'PA']

    all_metadata_list = []
    for _, row in tqdm(metadata.iterrows(), total=metadata.shape[0], desc='Iterating over mimic-cxr-2.0.0-metadata.csv.gz'):
        study_time = str(row['StudyTime']).split('.')[0]
        if len(study_time) < 5:
            study_time = '000000'   # if it's not possible to extract a valid hour, minute, second
        all_metadata_list.append(
            {
                "subject_id": int(row['subject_id']),
                "study_id": int(row['study_id']),
                "dicom_id": str(row['dicom_id']),
                "study_datetime": pd.to_datetime(str(row['StudyDate']) + ' ' + study_time, format='%Y%m%d %H%M%S').strftime('%Y-%m-%d %H:%M:%S'),
                "image_path": f'{input_path}/p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{int(row["study_id"])}/{row["dicom_id"]}.jpg',
            }
        )

    all_metadata_df = pd.DataFrame(all_metadata_list)
    all_metadata_df.set_index('subject_id', inplace=True)
    return all_metadata_df

def filter_cxr(all_metadata, all_stays_path):
    all_stays = pd.read_csv(all_stays_path)
    all_stays = all_stays[['subject_id','hadm_id','stay_id','admittime','dischtime']]
    all_stays.loc[:, 'admittime'] = pd.to_datetime(all_stays['admittime'], format='%Y-%m-%d %H:%M:%S')
    all_stays.loc[:, 'dischtime'] = pd.to_datetime(all_stays['dischtime'], format='%Y-%m-%d %H:%M:%S')
    all_stays = all_stays.groupby(['subject_id']).agg({'admittime': 'min', 'dischtime': 'max'}).reset_index()
    
    filtered_metadata = all_metadata.merge(all_stays, on='subject_id', how='inner')
    filtered_metadata['study_datetime'] = pd.to_datetime(filtered_metadata['study_datetime'], format='%Y-%m-%d %H:%M:%S')
    filtered_metadata = filtered_metadata[(filtered_metadata['study_datetime'] > (filtered_metadata['admittime'] - pd.Timedelta(days=30))) | (filtered_metadata['study_datetime'] < (filtered_metadata['dischtime'] + pd.Timedelta(days=30)))]
    filtered_metadata = filtered_metadata.sort_values('study_datetime').drop_duplicates(subset='subject_id', keep='first')
    return filtered_metadata[['subject_id','image_path']]

# CXR IMAGE LINKING #

def link_cxr(filtered_metadata, output_path):
    root_episodes = os.listdir(output_path)
    num_subjects = set()
    num_episodes = 0

    for _, row in tqdm(filtered_metadata.iterrows(), total=filtered_metadata.shape[0], desc="Linking subject-specific CXR images to episode-specific sub-dir."):
        matches = [ep for ep in root_episodes if int(ep.split('_')[0])==int(row['subject_id'])]
        if len(matches) > 0:
            for episode_match in matches:
                output_episode_dir = os.path.join(output_path, episode_match)
                try:
                    shutil.copy(row['image_path'], output_episode_dir)
                    num_episodes += 1
                    num_subjects.add(row['subject_id'])
                except FileNotFoundError:       # skip CXR images that are not found
                    continue
    
    print(f'{num_episodes} episodes of {len(num_subjects)} subjects have an asociated CXR image.')