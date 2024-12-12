import csv
import os
import pandas as pd

# READ DATATABLE CSVs INTO DATAFRAMES (FOR 1_extract_subjects.py) #

def read_patients_table(path):
    pats = pd.read_csv(path)
    columns = ['subject_id', 'gender', 'anchor_age', 'dod']  
    pats = pats[columns]
    pats.dod = pd.to_datetime(pats.dod)
    return pats

def read_admissions_table(path):
    admits = pd.read_csv(path)
    admits = admits[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime']]
    admits.admittime = pd.to_datetime(admits.admittime)
    admits.dischtime = pd.to_datetime(admits.dischtime)
    admits.deathtime = pd.to_datetime(admits.deathtime)
    return admits

def read_icustays_table(path):
    stays = pd.read_csv(path)
    stays.intime = pd.to_datetime(stays.intime)
    stays.outtime = pd.to_datetime(stays.outtime)
    return stays

def read_icd_diagnoses_table(path):
    codes = pd.read_csv(f'{path}/d_icd_diagnoses.csv')
    codes = codes[['icd_code', 'long_title']]

    diagnoses = pd.read_csv(f'{path}/diagnoses_icd.csv')
    diagnoses = diagnoses.merge(codes, how='inner', left_on='icd_code', right_on='icd_code')
    diagnoses[['subject_id', 'hadm_id', 'seq_num']] = diagnoses[['subject_id', 'hadm_id', 'seq_num']].astype(int)
    return diagnoses

def read_events_table_by_row(path, table):
    nb_rows = {'chartevents': 313645063, 'labevents': 118171367, 'outputevents': 4234967}  # for MIMIC-IV version 2.2 specifically
    csv_files = {'chartevents': 'icu/chartevents.csv', 'labevents': 'hosp/labevents.csv', 'outputevents': 'icu/outputevents.csv'}
    reader = csv.DictReader(open(os.path.join(path, csv_files[table.lower()]), 'r'))
    for i, row in enumerate(reader):
        if 'stay_id' not in row:
            row['stay_id'] = ''
        yield row, i, nb_rows[table.lower()]

# READ RELEVANT CSVs INTO DATAFRAMES FOR 3_extract_episodes.py #

def read_itemid_to_variable_map(path, variable_column='LEVEL2'):
    var_map = pd.read_csv(path).fillna('').astype(str)
    var_map.COUNT = var_map.COUNT.astype(int)
    var_map = var_map[(var_map[variable_column] != '') & (var_map.COUNT > 0)]
    var_map.ITEMID = var_map.ITEMID.astype(int)
    var_map = var_map[[variable_column, 'ITEMID', 'MIMIC LABEL']]
    var_map = var_map.rename({variable_column: 'variable', 'MIMIC LABEL': 'mimic_label'}, axis=1)
    var_map.columns = var_map.columns.str.lower()
    return var_map

def read_variable_ranges(path, variable_column='LEVEL2'):
    columns = [variable_column, 'OUTLIER LOW', 'VALID LOW', 'IMPUTE', 'VALID HIGH', 'OUTLIER HIGH']
    
    to_rename = dict(zip(columns, [c.replace(' ', '_') for c in columns]))
    to_rename[variable_column] = 'variable'

    var_ranges = pd.read_csv(path, index_col=None)
    var_ranges = var_ranges[columns]
    var_ranges.rename(to_rename, axis=1, inplace=True)
    var_ranges.set_index('variable', inplace=True)
    return var_ranges.loc[var_ranges.notnull().all(axis=1)]

def read_stays(subject_path):
    stays = pd.read_csv(os.path.join(subject_path, 'stays.csv'), index_col=None)
    stays.intime = pd.to_datetime(stays.intime)
    stays.outtime = pd.to_datetime(stays.outtime)
    stays.dod = pd.to_datetime(stays.dod)
    stays.deathtime = pd.to_datetime(stays.deathtime)
    stays.sort_values(by=['intime', 'outtime'], inplace=True)
    return stays

def read_diagnoses(subject_path):
    return pd.read_csv(os.path.join(subject_path, 'diagnoses.csv'), index_col=None)

def read_events(subject_path):
    events = pd.read_csv(os.path.join(subject_path, 'events.csv'), index_col=None)
    events = events[events.value.notnull()]
    events.charttime = pd.to_datetime(events.charttime)
    events.hadm_id = events.hadm_id.fillna(value=-1).astype(int)
    events.stay_id = events.stay_id.fillna(value=-1).astype(int)
    events.valuenum = events.valuenum.fillna('').astype(str)
    return events