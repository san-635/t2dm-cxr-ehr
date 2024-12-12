import os
import pandas as pd
import shutil
import argparse
from tqdm import tqdm

'''
This script moves the pairs of episode{#}.csv and episode{#}_timeseries.csv files 
in subject-specific sub-dir. of 'ehr_root' dir. to separate episode-specific sub-dir. in 'root' dir.
'''

def main():
    parser = argparse.ArgumentParser(description="4_regroup_episodes.py")
    parser.add_argument('--input_path', '-ip', type=str, help="'ehr_root' dir. containing subject-specific sub-dir.",
                        default='ehr_root')
    parser.add_argument('--output_path', '-op', type=str, help="'root' dir. where episode-specific sub-dir. are to be saved.",
                        default='root')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    
    subjects = [x for x in os.listdir(args.input_path) if x.isdigit()]
    for subject_id in tqdm(subjects, desc=f'Extracting episode-specific sub-dir.'):

        # CREATE A DICT WITH PAIRS OF {#} AND CORRESPONDING LIST OF episode{#}.csv AND episode{#}_timeseries.csv FILES #

        episode_files = {}
        for file in os.listdir(os.path.join(args.input_path, subject_id)):
            if "episode" in file:

                # extract episode number {#} from the filenames
                if "timeseries" in file:
                    episode_number = file.split('_')[0].replace('episode', '')
                else:
                    episode_number = file.split('.')[0].replace('episode', '')
                
                # create a key-value pair in episode_files for each unique {#}
                if episode_number not in episode_files:
                    episode_files[episode_number] = []
                
                # pair episode{#}.csv and episode{#}_timeseries.csv to the corresponding {#} key
                episode_files[episode_number].append(file)

        # MOVE PAIRS OF episode{#}.csv AND episode{#}_timeseries.csv FILES TO EPISODE-SPECIFIC SUB-DIR. #

        for episode_number, files in episode_files.items():
            if len(files) == 2:
                # find stay_id from non-timeseries file
                stay_id = None
                for file in files:
                    if "timeseries" not in file:
                        df = pd.read_csv(os.path.join(args.input_path, subject_id, file))
                        if df.shape[0] > 0:
                            stay_id = df["Icustay"].iloc[0]
                            break
                
                # move the files to episode-specific sub-dir. called {subject_id}_{stay_id} in 'root' dir.
                if stay_id:
                    episode_dir = os.path.join(args.output_path, f'{subject_id}_{stay_id}')
                    os.makedirs(episode_dir, exist_ok=True)
                    for file in files:
                        shutil.copy(os.path.join(args.input_path, subject_id, file), os.path.join(episode_dir, file))
                
                # note: subject-specific sub-dir. in 'ehr_root' dir. still contain stays.csv, diagnoses.csv, events.csv, and all episode{#}.csv and episode{#}_timeseries.csv files
    
if __name__ == '__main__':
    main()