import os
import random
import pandas as pd
import argparse
import glob
from tqdm import tqdm
random.seed(0)

'''
This script partitions episodes in 'root' dir. with all required modalities into train (70%), val (10%) and test (20%) sets
'''

# IDENTIFY EPISODES WITH RELEVANT ASSOCIATED MODALITIES #

def get_subdir(input_path, EHR_CXR_only=False):
    subdir = []
    for episode in tqdm(os.listdir(input_path), desc=f"Iterating over episodes in 'root'"):
        ecg_files = glob.glob(os.path.join(input_path, episode, '*ecg.csv'))
        cxr_files = glob.glob(os.path.join(input_path, episode, '*.jpg'))
        if not EHR_CXR_only:
            if (len(ecg_files)==1) and (len(cxr_files)==1):
                subdir.append(episode)
        else:
            if (len(cxr_files)==1):
                subdir.append(episode)

    return subdir

# PARTITION THESE EPISODES INTO TRAIN, VAL AND TEST #

def partition_subdir(subdir, train_ratio=0.7, val_ratio=0.1, EHR_CXR_only=False):
    # randomly shuffle the episodes and partition them
    random.shuffle(subdir)
    train_episodes = subdir[:int(len(subdir)*train_ratio)]
    val_episodes = subdir[int(len(subdir)*train_ratio):int(len(subdir)*(train_ratio+val_ratio))]
    test_episodes = subdir[int(len(subdir)*(train_ratio+val_ratio)):]

    assert len(set(train_episodes) & set(val_episodes)) == 0
    assert len(set(train_episodes) & set(test_episodes)) == 0
    assert len(set(val_episodes) & set(test_episodes)) == 0

    # create a df indicating the partition that each episode belongs to, and save it
    partition_list = [(episode, 'train') for episode in train_episodes] + \
                     [(episode, 'val') for episode in val_episodes] + \
                     [(episode, 'test') for episode in test_episodes]
    partition_df = pd.DataFrame(partition_list, columns=['<subject_id>_<stay_id>', 'partition'])

    if EHR_CXR_only:
        partition_df.to_csv(os.path.join(os.path.dirname(__file__), 'D_E+C_partitions.csv'), index=False)
    else:
        partition_df.to_csv(os.path.join(os.path.dirname(__file__), 'D_E+C+G_partitions.csv'), index=False)

    subjects = list(set([episode.split('_')[0] for episode in subdir]))
    print(f'{len(subjects)} unique subjects exist across all partitions of the dataset.')

def main():
    parser = argparse.ArgumentParser(description="9_partition_samples.py")
    parser.add_argument('--input_path', '-ip', type=str, help="'root' dir. containing episode-specific sub-dir.", default='root')
    parser.add_argument('--EHR_CXR_only', action='store_true', help="If creating D_E+C completely independent of ECG data.", default=False)
    args = parser.parse_args()

    subdir = get_subdir(args.input_path, EHR_CXR_only=args.EHR_CXR_only)
    partition_subdir(subdir, train_ratio=0.7, val_ratio=0.1, EHR_CXR_only=args.EHR_CXR_only)

if __name__ == '__main__':
    main()