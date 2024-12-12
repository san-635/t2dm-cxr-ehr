from __future__ import absolute_import
from __future__ import print_function
import os
import numpy as np
import torch

from fusions.arguments import args_parser

from fusions.fusion_trainer import EHR_ECG_CXR_FusionTrainer, EHR_CXR_FusionTrainer
from fusions.datamodule import EHR_ECG_CXR_DataModule

parser = args_parser()
args = parser.parse_args()

def main():
    # GPU settings #
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Set seeds for reproducibility #
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialise data loaders and logger #
    datamodule = EHR_ECG_CXR_DataModule(args)
    datamodule.setup()
    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()
    test_dl = datamodule.test_dataloader()

    os.makedirs(args.save_dir, exist_ok=True)
    with open(f"{args.save_dir}/args.txt", 'w') as results_file:
        for arg in vars(args):
            print(f"  {arg:<40}: {getattr(args, arg)}")
            results_file.write(f"  {arg:<40}: {getattr(args, arg)}\n")

    # Initialise appropriate trainer #
    if args.modalities == 'EHR+ECG+CXR':
        trainer = EHR_ECG_CXR_FusionTrainer(args, train_dl, val_dl, test_dl=test_dl)
    elif args.modalities == 'EHR+CXR':
        trainer = EHR_CXR_FusionTrainer(args, train_dl, val_dl, test_dl=test_dl)

    # Train or test #
    if args.mode == 'train':
        print("\nTraining & validation:")
        trainer.train()
    elif args.mode == 'eval':
        print("\nTesting:")
        trainer.eval()
    else:
        raise ValueError("No implementation for args.mode")

if __name__ == "__main__":
    main()