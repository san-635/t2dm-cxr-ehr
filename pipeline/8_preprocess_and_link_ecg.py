import argparse

from ecg_preprocessing import *

'''
This script preprocesses relevant ECG signals for subjects
and moves them to respective episode-specific sub-dir. in 'root' dir.
'''

parser = argparse.ArgumentParser(description="8_preprocess_and_link_ecg.py")
parser.add_argument('--input_path', '-ip', type=str, help="'ecg_root/ecg_orig' dir. containing extracted ECG signals.", default='ecg_root/ecg_orig')
parser.add_argument('--filtered_path', '-fp', type=str, help="'ecg_root/ecg_filtered' dir. where filtered ECG signals are to be stored.", default='ecg_root/ecg_filtered')
parser.add_argument('--preprocessed_path', '-pp', type=str, help="'ecg_root/ecg_preprocessed' dir. where preprocessed ECG signals are to be stored.", default='ecg_root/ecg_preprocessed')
parser.add_argument('--output_path', '-op', type=str, help="'root' dir. where preprocessed ECG signals are to be stored in respective episode-specific sub-dir.", default='root')
parser.add_argument('--delete', '-d', action='store_true', help="Delete dir. from previous step to save disk space.", default=True)
args, _ = parser.parse_known_args()

os.makedirs(args.filtered_path, exist_ok=True)
os.makedirs(args.preprocessed_path, exist_ok=True)

# ECG SIGNAL PREPROCESSING #

# step 1/2: apply 5th order Butterworth filter [0.5Hz, 150Hz] across all leads of the ECG signals
filter_ecg(args.input_path, args.filtered_path, delete_prev=args.delete)

# step 2/2: pool the ECG signals to create preprocessed ECG signals of shape (pool_to_size, 12)
pool_ecg(args.filtered_path, args.preprocessed_path, pool_to_size=100, delete_prev=args.delete)

# ECG SIGNAL LINKING #
# move the preprocessed subject-specific ECG signals to all corresponding episode-specific sub-dir. in 'root' dir.

link_ecg(args.preprocessed_path, args.output_path, delete_prev=args.delete)