import argparse
import os
import sys
from tqdm import tqdm

from ehr_readers import read_itemid_to_variable_map, read_variable_ranges, read_stays, read_diagnoses, read_events
from ehr_preprocessing import *

'''
This script extracts episodes (each with episode{#}.csv and episode{#}_timeseries.csv files) for each subject
and saves them to respective subject-specific sub-dir. in 'ehr_root' dir.
'''

parser = argparse.ArgumentParser(description="3_extract_episodes.py")
parser.add_argument('--input_path', '-ip', type=str, help="'ehr_root' dir. containing subject-specific sub-dir.",
                    default='ehr_root')
parser.add_argument('--variable_map_file', '-v', type=str, help="CSV containing relevant event variables that form the EHR features.",
                    default=os.path.join(os.path.dirname(__file__), 'ehr_itemid_to_variable_map.csv'))
parser.add_argument('--reference_range_file', '-r', type=str, help="CSV containing reference ranges for EHR features.",
                    default=os.path.join(os.path.dirname(__file__), 'ehr_variable_ranges.csv'))
args = parser.parse_args()

# READ RELEVANT CSVs FOR TIME SERIES EPISODIC DATA PREPROCESSING #

var_map = read_itemid_to_variable_map(args.variable_map_file)
variables = var_map.variable.unique()
ranges = read_variable_ranges(args.reference_range_file)

for subject_dir in tqdm(os.listdir(args.input_path), desc='Extracting episodic data for subjects'):
    dn = os.path.join(args.input_path, subject_dir)
    subject_id = int(subject_dir)

    try:
        if not os.path.isdir(dn):
            raise Exception
    except:
        continue

    try:
        stays = read_stays(os.path.join(args.input_path, subject_dir))
        diagnoses = read_diagnoses(os.path.join(args.input_path, subject_dir))
        events = read_events(os.path.join(args.input_path, subject_dir))
    except:
        sys.stderr.write(f'Error reading from disk for subject {subject_id}')
        continue

    # NON-TIME SERIES EPISODIC DATA PREPROCESSING #

    # step 1/1: extract stay_id, non-ts EHR features, height (nan), weight (nan), los, and separate binary columns for each T2DM-relevant diagnosis code
    non_timeseries = assemble_non_timeseries(stays, diagnoses)

    # TIME SERIES EPISODIC DATA PREPROCESSING #

    # step 1/5: ensure only those event variables that form the EHR features are kept
    events = map_itemids_to_variables(events, var_map)
    
    # step 2/5: remove outliers and clean measurements for each event variable
    for variable in variables:
        events = remove_outliers_for_variable(events, variable, ranges)
    events = clean_events(events)
    if events.shape[0] == 0:
        continue

    # step 3/5: extract event recorded times (charttime), stay_ids and the relevant event variables
    timeseries = convert_events_to_timeseries(events, variables=variables)

    for i in range(stays.shape[0]):
        stay_id = stays.stay_id.iloc[i]
        intime = stays.intime.iloc[i]
        outtime = stays.outtime.iloc[i]

        # step 4/5: break up into unique episodes, based on ICU stay intime and outtime
        episode = get_events_for_episode(timeseries, stay_id, intime, outtime)
        if episode.shape[0] == 0:
            continue

        # step 5/5: replace the event recorded times (charttime) with hours elapsed since intime
        episode = add_hours_elapsed_to_events(episode, intime).set_index('HOURS').sort_index(axis=0)

        # SAVE NON-TIME SERIES EPISODIC DATA TO RESPECTIVE SUBJECT-SPECIFIC SUB-DIR. AS 'episode{#}.csv' #
        
        # extract and store the first non-null weight and height for non-time series episodic data
        # can be removed if not needed for data analysis (not used as a non-time series EHR feature)
        if stay_id in non_timeseries.index:
            non_timeseries.loc[stay_id, 'Weight'] = get_first_valid_from_timeseries(episode, 'Weight')
            non_timeseries.loc[stay_id, 'Height'] = get_first_valid_from_timeseries(episode, 'Height')
        
        non_timeseries.loc[non_timeseries.index == stay_id].to_csv(os.path.join(args.input_path, subject_dir, f'episode{i+1}.csv'), index_label='Icustay')
        
        # SAVE TIME SERIES EPISODIC DATA TO RESPECTIVE SUBJECT-SPECIFIC SUB-DIR. AS 'episode{#}_timeseries.csv' #
        
        columns = list(episode.columns)
        columns_sorted = sorted(columns, key=(lambda x: "" if x == "Hours" else x))
        episode = episode[columns_sorted]
        episode.to_csv(os.path.join(args.input_path, subject_dir, f'episode{i+1}_timeseries.csv'), index_label='Hours')