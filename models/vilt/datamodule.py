import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from vilt.dataset import EHR_ECG_CXR_Dataset
from vilt.ehr_ecg_utils.utils import Discretizer, Normalizer

class EHR_ECG_CXR_DataModule(LightningDataModule):
    def __init__(self, _config):
        super().__init__()

        self.data_dir = _config["data_root"]
        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size
        
        self.max_text_len = _config["max_text_len"]-1       # -1 since [class] token is prepended later
        self.timestep = _config["timestep"]
        self.impute_strategy = _config["impute_strategy"]
        
        self.max_ecg_len = _config["max_ecg_len"]-1         # -1 since [class] token is prepended later

        self.image_size = _config["image_size"]
        self.train_transform_keys = (["default_train"] if len(_config["train_transform_keys"])==0 else _config["train_transform_keys"])
        self.val_transform_keys = (["default_val"] if len(_config["val_transform_keys"])==0 else _config["val_transform_keys"])

        self.noisy = False
        self.noise_level = 0.0
        self.image_noise_type = None
        self.missing = False
        self.missing_level = 0.0

        if _config["test_only"] == True:
            if "noise" in _config["ablation"]:
                self.noisy = True
                self.noise_level = _config["noise_level"]
                self.image_noise_type = _config["image_noise_type"]
            if "missing" in _config["ablation"]:
                self.missing = True
                self.missing_level = _config["missing_level"]

        self.setup_flag = False

    @property
    def dataset_cls(self):
        return EHR_ECG_CXR_Dataset
    
    # initialise the "train" instance of the dataset class
    def set_train_dataset(self):
        discretizer = Discretizer(timestep=self.timestep, impute_strategy=self.impute_strategy, store_masks=True, start_time='zero', config_path='vilt/ehr_ecg_utils/discretizer_config.json')
        ehr_normalizer = Normalizer()
        ecg_normalizer = Normalizer()
        
        if self.impute_strategy == 'mean':
            ehr_normalizer_state = 'ehr_normalizer__{}_{}h_zero'.format(self.max_text_len, self.timestep)
        else:
            ehr_normalizer_state = 'ehr_normalizer__{}_{}h_{}'.format(self.max_text_len, self.timestep, self.impute_strategy)
        
        ehr_normalizer_path = os.path.join(os.path.dirname(__file__), "ehr_ecg_utils", ehr_normalizer_state)
        if not os.path.exists(ehr_normalizer_path):
            raise FileNotFoundError(f"Pickle file not found at {ehr_normalizer_path}. Run normalizer_states.py script first.")
        ehr_normalizer.load_params(ehr_normalizer_path)

        ecg_normalizer_path = os.path.join(os.path.dirname(__file__), "ehr_ecg_utils", 'ecg_normalizer')
        if not os.path.exists(ecg_normalizer_path):
            raise FileNotFoundError(f"Pickle file not found at {ecg_normalizer_path}. Run normalizer_states.py script first.")
        ecg_normalizer.load_params(ecg_normalizer_path)
        
        self.train_dataset = self.dataset_cls(
            discretizer,
            ehr_normalizer,
            ecg_normalizer,
            split="train",
            dataset_dir=self.data_dir,
            max_text_len=self.max_text_len,
            max_ecg_len=self.max_ecg_len,
            transform_keys=self.train_transform_keys,
            image_size=self.image_size
        )

    # initialise the "val" instance of the dataset class
    def set_val_dataset(self):
        discretizer = Discretizer(timestep=self.timestep, impute_strategy=self.impute_strategy, store_masks=True, start_time='zero', config_path='vilt/ehr_ecg_utils/discretizer_config.json')
        ehr_normalizer = Normalizer()
        ecg_normalizer = Normalizer()
        
        if self.impute_strategy == 'mean':
            ehr_normalizer_state = 'ehr_normalizer__{}_{}h_zero'.format(self.max_text_len, self.timestep)
        else:
            ehr_normalizer_state = 'ehr_normalizer__{}_{}h_{}'.format(self.max_text_len, self.timestep, self.impute_strategy)
        ehr_normalizer_path = os.path.join(os.path.dirname(__file__), "ehr_ecg_utils", ehr_normalizer_state)
        ehr_normalizer.load_params(ehr_normalizer_path)

        ecg_normalizer_path = os.path.join(os.path.dirname(__file__), "ehr_ecg_utils", 'ecg_normalizer')
        ecg_normalizer.load_params(ecg_normalizer_path)

        self.val_dataset = self.dataset_cls(
            discretizer,
            ehr_normalizer,
            ecg_normalizer,
            split="val",
            dataset_dir=self.data_dir,
            max_text_len=self.max_text_len,
            max_ecg_len=self.max_ecg_len,
            transform_keys=self.val_transform_keys,
            image_size=self.image_size
        )
    
    # initialise the "test" instance of the dataset class
    def set_test_dataset(self):
        discretizer = Discretizer(timestep=self.timestep, impute_strategy=self.impute_strategy, store_masks=True, start_time='zero', config_path='vilt/ehr_ecg_utils/discretizer_config.json')
        ehr_normalizer = Normalizer()
        ecg_normalizer = Normalizer()
        
        if self.impute_strategy == 'mean':
            ehr_normalizer_state = 'ehr_normalizer__{}_{}h_zero'.format(self.max_text_len, self.timestep)
        else:
            ehr_normalizer_state = 'ehr_normalizer__{}_{}h_{}'.format(self.max_text_len, self.timestep, self.impute_strategy)
        ehr_normalizer_path = os.path.join(os.path.dirname(__file__), "ehr_ecg_utils", ehr_normalizer_state)
        ehr_normalizer.load_params(ehr_normalizer_path)

        ecg_normalizer_path = os.path.join(os.path.dirname(__file__), "ehr_ecg_utils", 'ecg_normalizer')
        ecg_normalizer.load_params(ecg_normalizer_path)

        self.test_dataset = self.dataset_cls(
            discretizer,
            ehr_normalizer,
            ecg_normalizer,
            split="test",
            dataset_dir=self.data_dir,
            max_text_len=self.max_text_len,
            max_ecg_len=self.max_ecg_len,
            transform_keys=self.val_transform_keys,
            image_size=self.image_size,
            noisy=self.noisy,
            noise_level=self.noise_level,
            image_noise_type=self.image_noise_type,
            missing=self.missing,
            missing_level=self.missing_level
        )

    # prepare_data() is not overridden as distributed training is not used

    # setup() is called before the first epoch
    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.setup_flag = True

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.my_collate,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.my_collate,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.my_collate,
        )
        return loader