import csv
import os
import pandas as pd
from tqdm import tqdm

from ehr_readers import read_events_table_by_row

# STAYS PREPROCESSING #

def remove_icustays_with_transfers(stays):
    stays = stays[(stays.first_careunit == stays.last_careunit)]
    return stays[['subject_id', 'hadm_id', 'stay_id', 'last_careunit', 'intime', 'outtime', 'los']]

def merge_on_subject_admission(table1, table2):
    return table1.merge(table2, how='inner', left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])

def merge_on_subject(table1, table2):
    return table1.merge(table2, how='inner', left_on=['subject_id'], right_on=['subject_id'])

def filter_admissions_on_nb_icustays(stays):
    to_keep = stays.groupby('hadm_id').count()[['stay_id']].reset_index()
    to_keep1 = to_keep[(to_keep.stay_id == 1)][['hadm_id']]
    stays_one = stays[stays['hadm_id'].isin(to_keep1['hadm_id'])][['stay_id']]
    to_keep2 = to_keep[(to_keep.stay_id > 1)][['hadm_id']]

    stays_more_than_max = stays[stays['hadm_id'].isin(to_keep2['hadm_id'])]
    stays_more_than_max = stays_more_than_max.loc[stays_more_than_max.groupby('hadm_id')['intime'].idxmin()]
    stays_more_than_max = stays_more_than_max[['stay_id']]
    
    to_keep_stays = stays_one.merge(stays_more_than_max, how='outer', left_on='stay_id', right_on='stay_id')
    stays = stays.merge(to_keep_stays, how='inner', left_on='stay_id', right_on='stay_id')
    return stays

# DIAGNOSES PREPROCESSING #

def filter_diagnoses_on_stays(diagnoses, stays):
    return diagnoses.merge(stays[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates(), how='inner',
                           left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])

# BREAK UP STAYS AND DIAGNOSES BY SUBJECT #

def break_up_stays_by_subject(stays, output_path, subjects=None):
    subjects = stays.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up stays by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        os.makedirs(dn, exist_ok=True)
        stays[stays.subject_id == subject_id].sort_values(by='intime').to_csv(os.path.join(dn, 'stays.csv'), index=False)

def break_up_diagnoses_by_subject(diagnoses, output_path, subjects=None):
    subjects = diagnoses.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up diagnoses by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        os.makedirs(dn, exist_ok=True)
        diagnoses[diagnoses.subject_id == subject_id].sort_values(by=['stay_id', 'seq_num']).to_csv(os.path.join(dn, 'diagnoses.csv'), index=False)

# BREAK UP MIMIC-IV EVENT DATATABLES BY SUBJECT #

def read_events_table_and_break_up_by_subject(mimic4_path, table, output_path, items_to_keep=None, subjects_to_keep=None):
    obs_header = ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'valuenum']
    if items_to_keep is not None:
        items_to_keep = set([str(s) for s in items_to_keep])
    if subjects_to_keep is not None:
        subjects_to_keep = set([str(s) for s in subjects_to_keep])

    class DataStats(object):
        def __init__(self):
            self.curr_subject_id = ''
            self.curr_obs = []

    data_stats = DataStats()

    # function to append rows from an event table to the CSV
    def write_current_observations():
        dn = os.path.join(output_path, str(data_stats.curr_subject_id))
        os.makedirs(dn, exist_ok=True)
        fn = os.path.join(dn, 'events.csv')
        if not os.path.exists(fn) or not os.path.isfile(fn):
            f = open(fn, 'w')
            f.write(','.join(obs_header) + '\n')
            f.close()
        w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
        w.writerows(data_stats.curr_obs)
        data_stats.curr_obs = []    # reset .curr_obs

    nb_rows_dict = {'chartevents': 313645063, 'labevents': 118171367, 'outputevents': 4234967}  # for MIMIC-IV version 2.2 specifically
    nb_rows = nb_rows_dict[table.lower()]

    # append only relevant rows of event tables to the CSV
    for row,row_no,_ in tqdm(read_events_table_by_row(mimic4_path,table),total=nb_rows,desc=f'Processing {table} table'):
        if (subjects_to_keep is not None) and (row['subject_id'] not in subjects_to_keep):
            continue
        if (items_to_keep is not None) and (row['itemid'] not in items_to_keep):
            continue
        row_out = {'subject_id': row['subject_id'],
                   'hadm_id': row['hadm_id'],
                   'stay_id': '' if 'stay_id' not in row else row['stay_id'],
                   'charttime': row['charttime'],
                   'itemid': row['itemid'],
                   'value': row['valuenum'] if table=='LABEVENTS' else row['value'],
                   'valuenum': row['valueuom']}
        if data_stats.curr_subject_id != '' and data_stats.curr_subject_id != row['subject_id']:
            write_current_observations()
        data_stats.curr_obs.append(row_out)
        data_stats.curr_subject_id = row['subject_id']

    if data_stats.curr_subject_id != '':
        write_current_observations()