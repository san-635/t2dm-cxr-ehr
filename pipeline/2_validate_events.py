import os
import argparse
import pandas as pd
from tqdm import tqdm

'''
This script validates events.csv files in all subject-specific sub-dir. of 'ehr_root' dir. by:
    (1) excluding events with missing/invalid hadm_id or invalid stay_id
    (2) recovering missing stay_ids of events using their hadm_ids
'''

def is_subject_folder(x):
    return str.isdigit(x)

def main():
    parser = argparse.ArgumentParser(description="2_validate_events.py")
    parser.add_argument('--input_path', '-ip', type=str, help="'ehr_root' dir. containing subject-specific sub-dir.",
                        default='ehr_root')
    args = parser.parse_args()

    n_events, empty_hadm, no_hadm_in_stay, no_icustay, recovered, could_not_recover, icustay_missing_in_stays = 0, 0, 0, 0, 0, 0, 0
    subdir = os.listdir(args.input_path)
    subjects = list(filter(is_subject_folder, subdir))

    # VALIDATE events.csv FILE IN EACH SUBJECT-SPECIFIC SUB-DIR. #

    for subject in tqdm(subjects, desc='Validating subjects'):
        stays_df = pd.read_csv(os.path.join(args.input_path, subject, 'stays.csv'))

        assert(not stays_df['stay_id'].isnull().any())
        assert(not stays_df['hadm_id'].isnull().any())
        assert(len(stays_df['stay_id'].unique()) == len(stays_df['stay_id']))
        assert(len(stays_df['hadm_id'].unique()) == len(stays_df['hadm_id']))

        events_df = pd.read_csv(os.path.join(args.input_path, subject, 'events.csv'))
        n_events += events_df.shape[0]

        # exclude events with missing hadm_id
        empty_hadm += events_df['hadm_id'].isnull().sum()
        events_df = events_df.dropna(subset=['hadm_id'])

        # merge events.csv with stays.csv to recover missing stay_ids of events
        merged_df = events_df.merge(stays_df, left_on=['hadm_id'], right_on=['hadm_id'],
                                    how='left', suffixes=['', '_r'], indicator=True)
        
        # exclude events whose hadm_id in events.csv does not appear in stays.csv, i.e., invalid hadm_id
        no_hadm_in_stay += (merged_df['_merge'] == 'left_only').sum()
        merged_df = merged_df[merged_df['_merge'] == 'both']

        # recover missing stay_id of events using hadm_id in stays.csv
        cur_no_icustay = merged_df['stay_id'].isnull().sum()
        no_icustay += cur_no_icustay
        merged_df.loc[:, 'stay_id'] = merged_df['stay_id'].fillna(merged_df['stay_id_r'])
        recovered += cur_no_icustay - merged_df['stay_id'].isnull().sum()
        
        # exclude events whose stay_id could not be recovered
        could_not_recover += merged_df['stay_id'].isnull().sum()
        merged_df = merged_df.dropna(subset=['stay_id'])

        # exclude events whose stay_id in events.csv does not appear in stays.csv, i.e., invalid stay_id
        icustay_missing_in_stays += (merged_df['stay_id'] != merged_df['stay_id_r']).sum()
        merged_df = merged_df[(merged_df['stay_id'] == merged_df['stay_id_r'])]

        # update events.csv file
        to_write = merged_df[['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'valuenum']]
        to_write.to_csv(os.path.join(args.input_path, subject, 'events.csv'), index=False)

    assert(could_not_recover == 0)

    # PRINT OVERALL STATS #
    print('Overall stats from validating events:')
    print(f'n_events: {n_events}')                                  # total events
    print(f'empty_hadm: {empty_hadm}')                              # events excluded as hadm_id is missing in events.csv
    print(f'no_hadm_in_stay: {no_hadm_in_stay}')                    # events excluded as the hadm_id in events.csv is not in stays.csv (i.e., invalid admission)
    print(f'no_icustay: {no_icustay}')                              # events with missing stay_id in events.csv
    print(f'recovered: {recovered}')                                # events with recovered stay_ids
    print(f'could_not_recover: {could_not_recover}')                # events whose stay_id could not be recovered (must be zero)
    print(f'icustay_missing_in_stays: {icustay_missing_in_stays}')  # events excluded as the stay_id in events.csv is not in stays.csv (i.e., invalid stay)

if __name__ == "__main__":
    main()