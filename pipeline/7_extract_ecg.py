import argparse
import os

from ecg_extraction import *

'''
This script extracts relevant ECG signals for subjects
and stores them in subject-specific sub-dir. in 'ecg_root/ecg_orig' dir.
'''

parser = argparse.ArgumentParser(description="7_extract_ecg.py")
parser.add_argument('--mimic_ecg_path', type=str, help='Base filename for accessing the MIMIC-IV-ECG v1.0 dataset via wfdb.',
                    default='mimic-iv-ecg/1.0/')
parser.add_argument('--mimic_ecg_records', type=str, help='URL for relevant metadata of the MIMIC-IV-ECG v1.0 dataset.',
                    default='https://physionet.org/content/mimic-iv-ecg/1.0/record_list.csv')
parser.add_argument('--output_path', '-op', type=str, help="'ecg_root/ecg_orig' dir. where extracted ECG signals are to be stored.",
                    default='ecg_root/ecg_orig')
parser.add_argument('--ehr_root_path', '-e', type=str, help="'ehr_root' dir. containing subject-specific sub-dir.",
                    default='ehr_root')
args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)

# ECG SIGNAL EXTRACTION #

# step 1/2: identify ECGs collected during subjects' hospital admissions 
filtered_metadata = filter_metadata(args.mimic_ecg_records, os.path.join(args.ehr_root_path, 'all_stays.csv'))
filtered_metadata.to_csv(os.path.join(args.output_path, 'filtered_metadata.csv'), index=False)

# step 2/2: extract only these ECGs (especially, only the first one if multiple exist for a subject) and store them in subject-specific sub-dir. in 'ecg_root/ecg_orig' dir.
extract_ecg(args.mimic_ecg_path, filtered_metadata, args.output_path)