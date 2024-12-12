import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import signal
import shutil

# ECG SIGNAL PREPROCESSING #

def filter_ecg(input_path, output_path, delete_prev=False):
    directories = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    for subject_id in tqdm(directories, total=len(directories), desc='Filtering ECG signals'):
        subject_dir = os.path.join(input_path, subject_id)
        for file in os.listdir(subject_dir):
            if file.endswith('.csv'):
                ecg_df = pd.read_csv(os.path.join(subject_dir, file), header=None, index_col=False)
                ecg = np.array(ecg_df)

                # create and apply Butterworth bandpass filter to the ECG signals
                filter = signal.butter(N=5, Wn=[0.5, 150], btype='bandpass', analog=False, output='sos', fs=500)
                for i in range(ecg.shape[1]):
                    filtered_ecg = signal.sosfilt(filter, ecg[:, i])    
                    ecg[:, i] = filtered_ecg
                ecg_df_filtered = pd.DataFrame(ecg)

                output_dir = os.path.join(output_path, subject_id)
                os.makedirs(output_dir, exist_ok=True)
                ecg_df_filtered.to_csv(os.path.join(output_dir, f'{file.split("o")[0]}f.csv'), index=False, header=False)
                break
    
    if delete_prev:                 # delete the input_path dir. after saving the filtered ECG signals to output_path
        shutil.rmtree(input_path)

def pool_ecg(input_path, output_path, pool_to_size=100, delete_prev=False):
    directories = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    for subject_id in tqdm(directories, total=len(directories), desc='Pooling ECG signals'):
        subject_dir = os.path.join(input_path, subject_id)
        for file in os.listdir(subject_dir):
            if file.endswith('.csv'):
                ecg_df = pd.read_csv(os.path.join(subject_dir, file), header=None, index_col=False)
                ecg = np.array(ecg_df)
                
                # average every pooled_length rows to pool the ECG signal
                n_rows, n_leads = ecg.shape
                pooled_length = n_rows // pool_to_size
                ecg_pooled = np.zeros((pool_to_size, n_leads))
                for i in range(pool_to_size):
                    ecg_pooled[i, :] = np.mean(ecg[(i*pooled_length):((i*pooled_length)+pooled_length), :], axis=0)
                ecg_df_final = pd.DataFrame(ecg_pooled)

                output_dir = os.path.join(output_path, subject_id)
                os.makedirs(output_dir, exist_ok=True)
                ecg_df_final.to_csv(os.path.join(output_dir, f'{file.split("f")[0]}ecg.csv'), index=False, header=False)
                break
    
    if delete_prev:                 # delete the input_path dir. after saving the pooled ECG signals to output_path
        shutil.rmtree(input_path)

# ECG SIGNAL LINKING #

def link_ecg(input_path, output_path, delete_prev=False):
    ecg_subjects = [int(x) for x in os.listdir(input_path)]
    root_episodes = os.listdir(output_path)

    num_subjects = set()
    num_episodes = 0

    # identify the corresponding episodes for each subject and move the ECG signals appropriately
    for ecg_subject in tqdm(ecg_subjects, total=len(ecg_subjects), desc="Linking subject-specific ECG signals to episode-specific sub-dir."):
        ecg_study = os.listdir(os.path.join(input_path, str(ecg_subject)))
        matches = [ep for ep in root_episodes if int(ep.split('_')[0]) == ecg_subject]
        if len(matches) > 0:
            for episode_match in matches:
                output_episode_dir = os.path.join(output_path, episode_match)
                shutil.copy(os.path.join(input_path, str(ecg_subject), ecg_study[0]), output_episode_dir)
                num_episodes += 1
                num_subjects.add(ecg_subject)

    if delete_prev:                 # delete the input_path dir. after moving the ECG signals to output_path
        shutil.rmtree(input_path)
    
    print(f'{num_episodes} episodes of {len(num_subjects)} subjects have an associated ECG signal.')