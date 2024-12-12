import os
import argparse
import pandas as pd
import random
import shutil
import csv
from tqdm import tqdm
import glob
random.seed(49297)

# return a generator that reads 'D_E+C+G_partitions.csv' in subsets of size 'subset_size'
def read_subset(partitions_path, subset_size=500):
    with open(partitions_path, "r") as partition:
        _ = partition.readline()  # skip header
        while True:
            subset = partition.readlines(subset_size)
            if not subset:
                break
            yield [row.strip().split(',') for row in subset]    # yield a list [{subject_id}_{stay_id}, partition]

# process each subset of episodes, preparing relevant files
def process_subset(args, subset, writer_train, writer_val, writer_test, eps=1e-6):
    for episode in subset:
        episode_path = os.path.join(args.input_path, str(episode[0]))
        ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(episode_path)))

        with open(os.path.join(episode_path, ts_files[0])) as ts_file:
            # 1/7: get length of stay (los) in hours from 'episode{#}.csv'
            non_ts_df = pd.read_csv(os.path.join(episode_path, ts_files[0].replace("_timeseries", "")))
            if non_ts_df.shape[0] == 0:
                continue                                        # exclude episode if corresponding 'episode{#}.csv' is empty
            los = 24.0 * non_ts_df.iloc[0]['Length of Stay']
            if pd.isnull(los):                                  # exclude episode if los is missing
                continue
            
            # 2/7: keep only those rows of 'episode{#}_timeseries.csv' that were recorded within the los
            ts_lines = ts_file.readlines()
            header = ts_lines[0]
            ts_lines = ts_lines[1:]
            event_times = [float(line.split(',')[0]) for line in ts_lines]
            ts_lines = [line for (line, t) in zip(ts_lines, event_times) if -eps < t < los + eps]
            if len(ts_lines) == 0:                              # exclude episode if no relevant events exist
                continue
            
            # 3/7: save the episode's time series EHR features as '{subject_id}_{stay_id}_timeseries.csv' in appropriate partition sub-dir. of 'D_E+C+G' dir.
            output_dir = os.path.join(args.output_path, str(episode[1]))
            output_ts_file = str(episode[0]) + "_timeseries.csv"
            with open(os.path.join(output_dir, output_ts_file), "w") as outfile:
                outfile.write(header)
                for line in ts_lines:
                    outfile.write(line)

            # 4/7: get label for this episode
            t2dm_label = 0
            for col in non_ts_df.columns:
                if col.startswith('Diagnosis'):
                    if non_ts_df[col].iloc[0] == 1:
                        t2dm_label = 1
                        break
            
            # 5/7: get the episode's stay_id and non-time series EHR features
            stay_id = non_ts_df.iloc[0]['Icustay']
            gender = non_ts_df.iloc[0]['Gender']
            age = non_ts_df.iloc[0]['Age']
            family_history = non_ts_df.iloc[0]['Family history']

            # 6/7: copy the episode's CXR image to episode-specific sub-dir. in 'CXR_images' dir. of 'D_E+C+G' dir.
            cxr_file = glob.glob(os.path.join(episode_path, '*.jpg'))
            output_cxr_dir = os.path.join(args.output_path, 'CXR_images', str(episode[0]))
            os.makedirs(output_cxr_dir, exist_ok=True)
            output_cxr_path = os.path.join(output_cxr_dir, os.path.basename(cxr_file[0]))
            shutil.copy(cxr_file[0], output_cxr_path)
            
            # 7/7: convert the episode's ECG signal to a 1D list
            ecg_file = glob.glob(os.path.join(episode_path, '*ecg.csv'))
            output_ecg_list = pd.read_csv(ecg_file[0], header=None, index_col=False).to_numpy().flatten().tolist()

            # write the episode's label, non-time series EHR features, CXR image's path and ECG data to the 'listfile.csv' in appropriate partition sub-dir. of 'D_E+C+G' dir.
            listfile_row = (output_ts_file, stay_id, t2dm_label, gender, age, family_history, output_cxr_path, output_ecg_list)
            if str(episode[1]) == 'train':
                writer_train.writerow(listfile_row)
            elif str(episode[1]) == 'val':
                writer_val.writerow(listfile_row)
            elif str(episode[1]) == 'test':
                writer_test.writerow(listfile_row)

def main():
    parser = argparse.ArgumentParser(description="10_create_D_E+C+G.py")
    parser.add_argument('--input_path', '-ip', type=str, help="'root' dir. containing episode-specific sub-dir.", default='root')
    parser.add_argument('--output_path', '-op', type=str, help="'D_E+C+G' dir. where the D_E+C+G dataset is to be stored.", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'D_E+C+G'))
    parser.add_argument('--partitions_path', type=str, help="CSV file containing the episodes' allocation to partitions.",
                        default=os.path.join(os.path.dirname(__file__), 'D_E+C+G_partitions.csv'))
    parser.add_argument('--subset_size', '-ss', type=int, help="Size of subset to process at a time.", default=500)
    args = parser.parse_args()

    # CREATE REQUIRED SUB-DIR. IN 'D_E+C+G' DIR. #
    for i in ['train', 'val', 'test']:
        os.makedirs(os.path.join(args.output_path, i), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'CXR_images'), exist_ok=True)

    # PROCESS EPISODES #

    listfile_header = "timeseries,stay_id,label,gender,age,family_history,cxr_path,ecg"

    with open(os.path.join(args.output_path, "train", "listfile.csv"), "a", newline='') as train_file, \
         open(os.path.join(args.output_path, "val", "listfile.csv"), "a", newline='') as val_file, \
         open(os.path.join(args.output_path, "test", "listfile.csv"), "a", newline='') as test_file:
        
        writer_train = csv.writer(train_file)
        writer_val = csv.writer(val_file)
        writer_test = csv.writer(test_file)

        # write header if 'listfile.csv' is empty
        if os.path.getsize(train_file.name) == 0:
            writer_train.writerow(listfile_header.split(','))
        if os.path.getsize(val_file.name) == 0:
            writer_val.writerow(listfile_header.split(','))
        if os.path.getsize(test_file.name) == 0:
            writer_test.writerow(listfile_header.split(','))

        # process each partitions' episodes in subsets to avoid memory issues
        for subset in tqdm(read_subset(args.partitions_path, args.subset_size)):
            process_subset(args, subset, writer_train, writer_val, writer_test)

    # APPROPRIATELY PREPARE THE 'listfile.csv' OF EACH PARTITION #

    # randomly shuffle the rows in train and val set 'listfile.csv'
    for partition in ['train', 'val']:
        with open(os.path.join(args.output_path, partition, "listfile.csv"), "r") as listfile:
            lines = listfile.readlines()
            random.shuffle(lines[1:])
        with open(os.path.join(args.output_path, partition, "listfile.csv"), "w") as listfile:
            listfile.writelines(lines)

    # sort the rows in test set 'listfile.csv'
    with open(os.path.join(args.output_path, "test", "listfile.csv"), "r") as listfile:
        lines = listfile.readlines()
        lines = [lines[0]] + sorted(lines[1:])
    with open(os.path.join(args.output_path, "test", "listfile.csv"), "w") as listfile:
        listfile.writelines(lines)

if __name__ == '__main__':
    main()