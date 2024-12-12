import argparse
import pandas as pd

from ehr_readers import read_patients_table, read_admissions_table, read_icustays_table, read_icd_diagnoses_table
from ehr_extraction import *

'''
This script extracts stays.csv, diagnoses.csv and events.csv files for each subject
and saves them to subject-specific sub-dir. in 'ehr_root' dir.
'''

parser = argparse.ArgumentParser(description="1_extract_subjects.py")
parser.add_argument('mimic_iv_path', type=str, help="Dir. containing the downloaded MIMIC-IV v2.2 dataset.")
parser.add_argument('--output_path', '-op', type=str, help="'ehr_root' dir. where subject-specific sub-dir. are to be saved.",
                    default='ehr_root')
parser.add_argument('--event_tables', '-e', type=str, nargs='+', help="List of MIMIC-IV event datatables.",
                    default=['OUTPUTEVENTS', 'CHARTEVENTS', 'LABEVENTS'])
parser.add_argument('--itemids_file', '-i', type=str, help="CSV containing relevant event variables that form the EHR features.",
                    default=os.path.join(os.path.dirname(__file__), 'ehr_itemid_to_variable_map.csv'))
args, _ = parser.parse_known_args()

os.makedirs(args.output_path, exists_ok=True)

# READ RELEVANT MIMIC-IV DATATABLES INTO DATAFRAMES #

patients = read_patients_table(f'{args.mimic_iv_path}/hosp/patients.csv')
admits = read_admissions_table(f'{args.mimic_iv_path}/hosp/admissions.csv')
stays = read_icustays_table(f'{args.mimic_iv_path}/icu/icustays.csv')
diagnoses = read_icd_diagnoses_table(f'{args.mimic_iv_path}/hosp')
print(f'{stays.hadm_id.unique().shape[0]} hospital admissions with {stays.stay_id.unique().shape[0]} ICU stays exist for {stays.subject_id.unique().shape[0]} subjects in MIMIC-IV v2.2.')

# STAYS PREPROCESSING #

# step 1/3: exclude stays with transfers
stays = remove_icustays_with_transfers(stays)

# step 2/3: exclude admissions with no stays
stays = merge_on_subject_admission(stays, admits)
stays = merge_on_subject(stays, patients)

# step 3/3: keep only the first stay for admissions with multiple stays
# (i.e., each admission has only one stay, thus stay_id alone can be used as unique identifier for the episodes extracted later)
stays = filter_admissions_on_nb_icustays(stays)
print(f'{stays.hadm_id.unique().shape[0]} hospital admissions with {stays.stay_id.unique().shape[0]} ICU stays exist for {stays.subject_id.unique().shape[0]} subjects after preliminary preprocessing of MIMIC-IV v2.2.')

stays.to_csv(os.path.join(args.output_path, 'all_stays.csv'), index=False)

# DIAGNOSES PREPROCESSING #

# step 1/1: exclude diagnoses of excluded stays
diagnoses = filter_diagnoses_on_stays(diagnoses, stays)
diagnoses.to_csv(os.path.join(args.output_path, 'all_diagnoses.csv'), index=False)

# BREAK UP STAYS AND DIAGNOSES BY SUBJECT AND SAVE TO RESPECTIVE SUB-DIR. AS 'stays.csv' AND 'diagnoses.csv' #

subjects = stays.subject_id.unique()
break_up_stays_by_subject(stays, args.output_path, subjects=subjects)
break_up_diagnoses_by_subject(diagnoses, args.output_path, subjects=subjects)

# BREAK UP MIMIC-IV EVENT DATATABLES BY SUBJECT AND SAVE TO RESPECTIVE SUB-DIR. AS 'events.csv' #

items_to_keep = set([int(itemid) for itemid in pd.read_csv(args.itemids_file)['ITEMID'].unique()]) if args.itemids_file else None
for table in args.event_tables:
    read_events_table_and_break_up_by_subject(f'{args.mimic_iv_path}', table, args.output_path, items_to_keep=items_to_keep, subjects_to_keep=subjects)