import argparse
import os
import gzip
import pandas as pd

from cxr_preprocessing import *

'''
This script extracts and preprocesses relevant CXR images for subjects
and moves them to respective episode-specific sub-dir. in 'root' dir.
'''

parser = argparse.ArgumentParser(description="6_preprocess_and_link_cxr.py")
parser.add_argument('mimic_cxr_path', type=str, help="Dir. containing the downloaded MIMIC-CXR-JPG v2.0.0 dataset.")
parser.add_argument('--input_path', '-ip', type=str, help="'cxr_root' dir. containing resized CXR images.",
                    default='cxr_root')
parser.add_argument('--output_path', '-op', type=str, help="'root' dir. where preprocessed CXR images are to be stored in respective episode-specific sub-dir.",
                    default='root')
parser.add_argument('--ehr_root_path', '-e', type=str, help="'ehr_root' dir. containing subject-specific sub-dir.",
                    default='ehr_root')
args = parser.parse_args()

# CXR IMAGE PREPROCESSING #

metadata_file = os.path.join(args.mimic_cxr_path, 'mimic-cxr-2.0.0-metadata.csv.gz')
try:
    with gzip.open(metadata_file) as f:
        metadata_df = pd.read_csv(f)
except:
    raise FileNotFoundError(f"File {metadata_file} not found in {args.mimic_cxr_path}. Please download it from MIMIC-CXR-JPG v2.0.0 and save it there.")

# step 1/2: keep only PA frontal CXR images and extract relevant metadata
all_metadata = read_metadata(metadata_df, args.input_path)
print(f'{all_metadata.shape[0]} PA CXR studies exist for {len(set(all_metadata["subject_id"]))} subjects in MIMIC-CXR-JPG v2.0.0.')
all_metadata.to_csv(os.path.join(args.input_path, 'all_metadata.csv'), index=True)

# step 2/2: exclude CXR images taken beyond a month of the earliest admission time or latest discharge time
# and keep only the first CXR image if multiple images exist for a subject
filtered_metadata = filter_cxr(all_metadata, os.path.join(args.ehr_root_path,'all_stays.csv'))
filtered_metadata.to_csv(os.path.join(args.input_path, 'filtered_metadata.csv'), index=False)

# CXR IMAGE LINKING #
# move the preprocessed subject-specific CXR images to all corresponding episode-specific sub-dir. in 'root' dir.

link_cxr(filtered_metadata, args.output_path)