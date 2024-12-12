import argparse
import glob
import os
from PIL import Image, ImageFile
from multiprocessing.dummy import Pool
from tqdm import tqdm

'''
This script resizes the CXR images in MIMIC-CXR-JPG to the required dimensions and saves them in 'cxr_root' dir.
'''

parser = argparse.ArgumentParser(description="5_resize_cxr.py")
parser.add_argument('mimic_cxr_path', type=str, help="Dir. containing the downloaded MIMIC-CXR-JPG v2.0.0 dataset.")
parser.add_argument('--output_path', '-op', type=str, help="'cxr_root' dir. where all resized MIMIC-CXR-JPG images are to be stored.",
                    default='cxr_root')
args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# IDENTIFY CXR IMAGES TO BE RESIZED #

paths_all = glob.glob(f'{args.mimic_cxr_path}/**/*.jpg', recursive = True)
paths_resized = glob.glob(f'{args.output_path}/**/*.jpg', recursive = True)
resized_files = [os.path.basename(path) for path in paths_resized]
paths = [path for path in paths_all if os.path.basename(path) not in resized_files]

def resize_images(args, path):
    # extract the filename and path in order to use the same dir. structure as MIMIC-CXR-JPG dataset
    filename = path.split('/')[-1]                                                          # {dicom_id}.jpg
    filepath = path.split('/')[-4] + '/' + path.split('/')[-3] + '/' + path.split('/')[-2]  # pXX/pYYYYYYYY/sZZZZZZZZ

    img = Image.open(path)
    w, h = img.size

    # RESIZE IMAGES #

    # resize shorter side to 384, if longer side > 640, resize longer side to 640 and adjust shorter side accordingly
    if w < h:
        w_new = 384
        h_new = int(float(h)*float(w_new/w))
        if h_new > 640:
            h_new = 640
            w_new = int(float(w)*float(h_new/h))
    else:
        h_new = 384
        w_new = int(float(w)*float(h_new/h))
        if w_new > 640:
            w_new = 640
            h_new = int(float(h)*float(w_new/w))
    img = img.resize((w_new,h_new))
    
    # CREATE SAME DIR. STRUCTURE AS THE MIMIC-CXR-JPG DATASET AND SAVE RESIZED IMAGES APPROPRIATELY #

    output_dir = f'{args.output_path}/{filepath}'
    os.makedirs(output_dir, exist_ok=True)
    img.save(f'{output_dir}/{filename}')

# THREADING #

threads = 10
for i in tqdm(range(0, len(paths), threads)):
    paths_subset = paths[i : i+threads]
    pool = Pool(len(paths_subset))
    pool.map(lambda path: resize_images(args, path), paths_subset)
    pool.close()
    pool.join()