import os
import numpy as np
from PIL import Image
import torch
import re

from torch.utils.data import Dataset
from vilt.transforms import keys_to_transforms

class EHR_ECG_CXR_Dataset(Dataset):
    def __init__(self, discretizer, ehr_normalizer, ecg_normalizer, split, dataset_dir, max_text_len, max_ecg_len, transform_keys, image_size, noisy=False, noise_level=0.0, image_noise_type=None, missing=False, missing_level=0.0):
        self.discretizer = discretizer
        self.ehr_normalizer = ehr_normalizer
        self.ecg_normalizer = ecg_normalizer
        assert split in ["train", "val", "test"]
        self.split = split
        self._dataset_dir = dataset_dir
        self.max_text_len = max_text_len
        self.max_ecg_len = max_ecg_len
        self.transforms = keys_to_transforms(transform_keys, size=image_size)
        self.noisy = noisy
        self.noise_level = noise_level
        self.image_noise_type = image_noise_type
        self.missing = missing
        self.missing_level = missing_level
        
        listfile_path = os.path.join(dataset_dir, split, "listfile.csv")
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._data = [re.sub(r'(\[.*?\])', lambda x: x.group(1).replace(',', '|'), line) for line in self._data]  # replace commas in ECG signal with '|' to avoid splitting signal
        self._data = self._data[1:]
        self._data = [line.split(',') for line in self._data]
        self.data_map = {
            mas[0]: {
                'stay_id': int(float(mas[1])),
                'label': int(mas[2]),
                'gender': int(float(mas[3])),
                'age': float(mas[4]),
                'family_history': int(float(mas[5])),
                'cxr_path': mas[6],
                'ecg': np.array([float(i.strip("'")) for i in mas[7].split('\n')[0].strip('"').strip('[]').split('| ')]).reshape(self.max_ecg_len, 12)
            } for mas in self._data
        }
        self.names = list(self.data_map.keys())
    
    def read_timeseries(self, ts_filename):        
        ret = []
        with open(os.path.join(self._dataset_dir, self.split, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)
    
    def read_by_ts_filename(self, index):
        if isinstance(index, int):
            index = self.names[index]
        (X, header) = self.read_timeseries(index)
        ret = self.data_map[index]
        ret.update({'ts_filename': index, 'ts_rows': X, 'ts_header': header})
        return ret
    
    def get_text(self, index):
        if isinstance(index, int):
            index = self.names[index]
        ret = self.read_by_ts_filename(index)
        ts_data = ret["ts_rows"]
        non_ts_data = {"gender": ret["gender"], "age": ret["age"], "family_history": ret["family_history"]}
        
        # discretised + normalised timeseries and static EHR data of this sample and its mask
        ehr_data, t_mask = self.discretizer.transform(ts_data, non_ts_data, max_rows=self.max_text_len)
        ehr_data = self.ehr_normalizer.transform(ehr_data)
        
        # normalised ECG signal and its mask
        ecg_data = self.ecg_normalizer.transform(ret["ecg"])
        ecg_data = torch.tensor(ecg_data, dtype=torch.float32)
        e_mask = torch.ones(ecg_data.shape[0], dtype=torch.float32)

        label = int(ret["label"])
        cxr_path = ret["cxr_path"]
        return torch.tensor(ehr_data, dtype=torch.float32), torch.tensor(t_mask, dtype=torch.float32), ecg_data, e_mask, label, cxr_path

    def get_image(self, image_path):
        assert isinstance(image_path, str)
        image = Image.open(image_path).convert('RGB')
        image_tensor = [tr(image) for tr in self.transforms]
        return image_tensor

    def __getitem__(self, index):
        ehr_data, t_mask, ecg_data, e_mask, label, cxr_path = self.get_text(index)
        cxr_tensor = self.get_image(cxr_path)
        return {
            "image": cxr_tensor,    # a list with a single 3D tensor (shape: 3, H, W)
            "text": ehr_data,       # a 2D tensor (shape: no. of bins, ehr_n_var)
            "t_mask": t_mask,       # a 1D tensor (shape: no. of bins)
            "ecg": ecg_data,        # a 2D tensor (shape: max_ecg_len-1, 12)
            "e_mask": e_mask,       # a 1D tensor (shape: max_ecg_len-1)
            "label": label,         # a scalar (0 or 1)
        }

    def __len__(self):
        return len(self.names)

    def my_collate(self, batch):
        # RE-STUCTURE BATCH #
        batch_size = len(batch)
        keys = set([key for sample in batch for key in sample.keys()])
        dict_batch = {key: [sample[key] for sample in batch] for key in keys}

        # IMAGE MODALITY - PAD SMALLER IMAGES #
        img = dict_batch["image"]

        img_sizes = list()
        img_sizes += [ii.shape for i in img if i is not None for ii in i]
        for size in img_sizes:
            assert (len(size) == 3), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        max_height = max([i[1] for i in img_sizes])
        max_width = max([i[2] for i in img_sizes])

        view_size = len(img[0])
        new_images = [torch.zeros(batch_size, 3, max_height, max_width) for _ in range(view_size)]

        for bi in range(batch_size):
            orig_batch = img[bi]
            for vi in range(view_size):
                if orig_batch is None:
                    new_images[vi][bi] = None
                else:
                    orig = img[bi][vi]
                    # make appropriate changes to images for "noisy inputs" ablation study
                    if self.noisy == True:
                        orig = self.add_image_noise(orig, type=self.image_noise_type, noise_level=self.noise_level)
                    new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

        # make appropriate changes to images for "missing CXR modality" ablation study
        if self.missing == True:
            new_images = [self.apply_missing_image(new_images[0], missing_level=self.missing_level)]

        dict_batch["image"] = new_images

        # TEXT MODALITY - PAD EHR DATA AND MASK TENSORS #
        ehr_data = dict_batch["text"]
        ehr_np = [d.numpy() for d in ehr_data]
        ehr_padded = self.pad_timeseries(ehr_np)
        # make appropriate changes to EHR data for "noisy inputs" ablation study
        if self.noisy == True:
            ehr_tensor = torch.stack(ehr_padded)
            new_ehr = self.add_timeseries_noise(ehr_tensor, noise_level=self.noise_level)
        else:
            new_ehr = torch.stack(ehr_padded)
        dict_batch["text"] = new_ehr

        t_mask = dict_batch["t_mask"]
        mask_np = [np.array(m) for m in t_mask]
        mask_padded = self.pad_timeseries_mask(mask_np)
        mask_tensor = torch.stack(mask_padded)
        dict_batch["t_mask"] = mask_tensor

        # ECG MODALITY - NO PADDING REQUIRED FOR ECG SIGNAL OR ITS MASK #
        ecg_tensor = torch.stack(dict_batch["ecg"])
        # make appropriate changes to ECG data for "noisy inputs" ablation study
        if self.noisy == True:
            new_ecg = self.add_timeseries_noise(ecg_tensor, noise_level=self.noise_level)
        else:
            new_ecg = ecg_tensor
        dict_batch["ecg"] = new_ecg

        e_mask_tensor = torch.stack(dict_batch["e_mask"])
        dict_batch["e_mask"] = e_mask_tensor

        # batch["label"] is unchanged
        
        return dict_batch

    def add_timeseries_noise(self, batch, noise_level=0.1):
        dev = batch.device
        padding_mask = (batch != -100).float().to(dev)

        gaussian_noise = (torch.randn(batch.size(), device=dev) * noise_level).to(dev)
        uniform_noise = ((torch.rand(batch.size(), device=dev) - 0.5) * 2 * noise_level).to(dev)

        unpadded_batch_abs = torch.abs(batch) * padding_mask
        poisson_noise = (torch.poisson(unpadded_batch_abs * noise_level) - unpadded_batch_abs * noise_level).to(dev)    # compute Poisson noise using only non-padding values
        noisy_batch = batch + (gaussian_noise + uniform_noise + poisson_noise) * padding_mask
        return noisy_batch*padding_mask
    
    def add_image_noise(self, image, type, noise_level=0.1):
        noisy_image = image.clone()
        if 'gaussian' in type:
            gaussian_noise = torch.randn(image.size()) * noise_level
            noisy_image = torch.clamp(noisy_image + gaussian_noise, -1, 1)
        if 'salt_and_pepper' in type:
            _, height, width = image.size()
            num_salt = np.ceil(0.5 * height * width * noise_level)
            num_pepper = np.ceil(0.5 * height * width * noise_level)
            s_coords = [torch.randint(0, x, (int(num_salt),)) for x in image.shape]
            noisy_image[:, s_coords[1], s_coords[2]] = 1
            p_coords = [torch.randint(0, x, (int(num_pepper),)) for x in image.shape]
            noisy_image[:, p_coords[1], p_coords[2]] = -1
        if 'poisson' in type:
            vals = torch.tensor(len(torch.unique(noisy_image))).float()
            vals = 2 ** torch.ceil(torch.log2(vals))
            noisy_image = (torch.poisson((noisy_image + 1) * vals) / vals) - 1
            noisy_image = torch.clamp(noisy_image, -1, 1)
        return noisy_image

    def apply_missing_image(self, batch, missing_level=0.3):
        indices = list(range(batch.size(0)))
        np.random.shuffle(indices)
        missing_indices = indices[:int(batch.size(0) * missing_level)]
        missing_batch = batch.clone()
        for i in missing_indices:
            missing_batch[i] = torch.zeros_like(batch[i])
        return missing_batch

    def pad_timeseries(self, data):
        dtype = data[0].dtype
        padded_data = [np.concatenate([x, np.full((self.max_text_len - x.shape[0],) + x.shape[1:], -100, dtype=dtype)], axis=0) for x in data]
        return [torch.tensor(padded) for padded in padded_data]
    
    def pad_timeseries_mask(self, t_mask):
        dtype = t_mask[0].dtype
        padded_mask = [np.concatenate([x, np.zeros(self.max_text_len - x.shape[0], dtype=dtype)]) for x in t_mask]
        return [torch.tensor(padded) for padded in padded_mask]