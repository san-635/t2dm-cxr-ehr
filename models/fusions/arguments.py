import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='Arguments for ResNet-LSTM model')

    parser.add_argument('--gpu_device', type=int, default=0, help='GPU Device ID')
    parser.add_argument('--seed', type=int, default=0, help='Fixed seed for reproducibility')
    parser.add_argument('--modalities', type=str, default='EHR+ECG+CXR', help='Input modalities: EHR+ECG+CXR or EHR+CXR')
    parser.add_argument('--mode', type=str, default='train', help='Finetune modes: train or eval (only train mode is to be used with pretrain_ stages)')
    parser.add_argument('--stage', type=str, default='pretrain_ehr', help='Stages: pretrain_ehr, pretrain_ecg, pretrain_cxr, finetune_early or finetune_joint')
    parser.add_argument('--data_root', type=str, default='../D_E+C+G', help='Directory containing the dataset')
    parser.add_argument('--save_dir', type=str, help='Directory where output files are to be stored')

    parser.add_argument('--num_classes', type=int, default=1, help='Number of classes for classification')
    parser.add_argument('--dim', type=int, default=256, help='Number of hidden units in LSTM')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate for LSTM')
    parser.add_argument('--layers', type=int, default=2, help='Number of stacked LSTM modules: 1 or 2')
    
    parser.add_argument('--ehr_n_var', type=int, default=11, help='Number of EHR features (timeseries + static)')
    parser.add_argument('--max_text_len', type=int, default=96, help='Number of EHR tokens (i.e. bins) after padding')
    parser.add_argument('--timestep', type=float, default=0.5, help='Time distance between consecutive EHR tokens')
    parser.add_argument('--impute_strategy', type=str, default='zero', help='Imputation strategy for missing EHR timeseries features')
    
    parser.add_argument('--ecg_n_var', type=int, default=12, help='Number of ECG features (i.e. leads)')
    parser.add_argument('--max_ecg_len', type=int, default=100, help='Number of ECG tokens')

    parser.add_argument('--vision_backbone', type=str, default='resnet50', help='Variant of ResNet model: resnet18 or resnet34 or resnet50')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='Load pretrained ResNet model (False for "lack of pretraining" ablation study)')
    parser.add_argument('--train_transform_keys', type=str, default=['resnet'], help='Training image augmentations')
    parser.add_argument('--val_transform_keys', type=str, default=['resnet'], help='Val. and eval image augmentations')

    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--max_epoch', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Number of val epochs to wait for an improvement in val AUROC before early stopping')
    parser.add_argument('--beta_1', type=float, default=0.9, help='beta_1 param for Adam optimizer')

    parser.add_argument('--load_state_ehr', type=str, default=None, help='EHR encoder load state path (for finetune_early with train mode)')
    parser.add_argument('--load_state_ecg', type=str, default=None, help='ECG encoder load state path (for finetune_early with train mode)')
    parser.add_argument('--load_state_cxr', type=str, default=None, help='CXR encoder load state path (for finetune_early with train mode)')
    parser.add_argument('--load_state', type=str, default=None, help='Model load state path (for finetune_early or finetune_joint with eval mode)')

    parser.add_argument('--ablation', type=str, default='none', help='Ablation study: none or noise or missing')
    parser.add_argument('--noise_level', type=float, default=0.1, help='Noise level for "noisy inputs" ablation study: 0.1, 0.5, 0.7')
    parser.add_argument('--image_noise_type', type=str, default='gaussian+salt_and_pepper+poisson', help='Noise type in image modality for "noisy inputs" ablation study')
    parser.add_argument('--missing_level', type=float, default=0.3, help='Missing ratio for "missing CXR modality" ablation study: 0.3, 0.5, 0.7')
    
    return parser