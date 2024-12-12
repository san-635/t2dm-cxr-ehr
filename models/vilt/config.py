from sacred import Experiment
ex = Experiment("ViLT")


def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "mppd": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
        "mmimdb": 0,
        "hatememes": 0,
        "food101": 0,

        "ehr_ecg_cxr": 0,   # added for this study
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "vilt"
    seed = 0                # for reproducibility
    datasets = []
    loss_names = _loss_names({})
    batch_size = 0
    per_gpu_batchsize = 0
    data_root = ""
    log_dir = ""
    num_gpus = 1
    gpu_device = 0
    num_nodes = 1
    num_workers = 4         # reduce if running into a memory error
    precision = 16
    test_ratio = None
    test_type = None
    test_exp_name = None

    # Text Setting
    vqav2_label_size = 3129
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    # Multimodal Setting (added for this study)
    modalities = ""
    
    # EHR Setting (added for this study)
    ehr_n_var = 0
    max_text_len = 0
    timestep = 0
    impute_strategy = "zero"

    # ECG Setting (added for this study)
    max_ecg_len = 0
    
    # Image setting
    train_transform_keys = []
    val_transform_keys = []
    image_size = 0
    max_image_len = -1
    patch_size = 32
    draw_false_image = 0
    image_only = False

    # Transformer Setting
    vit = "vit_base_patch32_384"
    load_path = ""
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    warmup_steps = 2500
    decay_power = 1
    weight_decay = 0.01
    max_epoch = 100
    end_lr = 0
    lr_mult = 1

    # Downstream Setting
    get_recall_metric = False
    mmimdb_class_num = 23
    hatememes_class_num = 2
    food101_class_num = 101

    # Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    finetune_first = False

    # Ablation Study Setting (added for this study)
    ablation = ""
    noise_level = 0.0
    image_noise_type = ""
    missing_level = 0.0

# Configurations for this study
@ex.named_config
def task_finetune_ehr_ecg_cxr():
    load_path = "./vilt/vilt_200k_mlm_itm.ckpt"     # path to .ckpt file containing pre-trained ViLT weights
    exp_name = "finetune_ehr_ecg_cxr"
    loss_names = _loss_names({"ehr_ecg_cxr": 1})
    data_root = "../D_E+C+G"                        # path to directory containing the dataset
    log_dir = "./results/vilt/D_E+C+G"              # path to directory where results are to be saved

    learning_rate = 1e-4
    warmup_steps = 0.1
    decay_power = 1
    weight_decay = 2e-2
    optim_type = "adamw"

    batch_size = 256
    per_gpu_batchsize = 256
    max_epoch = 20
    val_check_interval = 0.5
    
    modalities = "EHR+ECG+CXR"                      # input modalities: EHR+ECG+CXR or EHR+CXR

    ehr_n_var = 11                                  # number of EHR features (timeseries + static)
    max_text_len = 97                               # (48h / 0.5h) + 1 where 0.5h=sampling rate and 48h=sampling duration
    timestep = 0.5                                  # sampling rate for EHR features
    impute_strategy = "zero"                        # imputation strategy for missing EHR features

    max_ecg_len = 101                               # number of ECG signal data points + 1

    train_transform_keys = ["pixelbert"]            # CXR image augmentation for training
    val_transform_keys = ["pixelbert"]              # CXR image augmentation for val
    image_size = 384                                # shorter size of CXR images

    ablation = "none"                               # "none" or "noise" for "noisy inputs" or "missing" for "missing CXR modality" ablation studies
    noise_level = 0.0                               # noise level for "noisy inputs" ablation study: 0.1, 0.5, 0.7
    image_noise_type = "gaussian__salt_and_pepper__poisson" # noise type in image modality for "noisy inputs" ablation study
    missing_level = 0.0                             # missing ratio for "missing CXR modality" ablation study: 0.3, 0.5, 0.7



@ex.named_config
def env_dandelin():
    data_root = "/data2/dsets/dataset"
    log_dir = "/data2/vilt/result"
    num_gpus = 8
    num_nodes = 1

@ex.named_config
def task_mlm_itm():
    exp_name = "mlm_itm"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


@ex.named_config
def task_mlm_itm_randaug():
    exp_name = "mlm_itm_randaug"
    datasets = ["coco", "vg", "sbu", "gcc"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


@ex.named_config
def task_mlm_itm_mpp():
    exp_name = "mlm_itm_mpp"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "mpp": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


@ex.named_config
def task_finetune_nlvr2():
    exp_name = "finetune_nlvr2"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4


@ex.named_config
def task_finetune_nlvr2_randaug():
    exp_name = "finetune_nlvr2_randaug"
    datasets = ["nlvr2"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4


@ex.named_config
def task_finetune_vqa():
    exp_name = "finetune_vqa"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10


@ex.named_config
def task_finetune_vqa_randaug():
    exp_name = "finetune_vqa_randaug"
    datasets = ["vqa"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

@ex.named_config
def task_finetune_hatememes():
    exp_name = "finetune_hatememes"
    datasets = ["Hatefull_Memes"]
    loss_names = _loss_names({"hatememes": 1})
    batch_size = 256
    max_epoch = 20
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-2 
    val_check_interval = 0.11
    weight_decay = 2e-2
#     optim_type = "adam"
    max_text_len = 128 
    
@ex.named_config
def task_finetune_food101():
    exp_name = "finetune_food101"
    datasets = ["Food101"]
    loss_names = _loss_names({"food101": 1})
    batch_size = 256
    max_epoch = 20
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-2
    val_check_interval = 0.2
    weight_decay = 2e-2
#     optim_type = "adam"
    max_text_len = 512     
    
@ex.named_config
def task_finetune_mmimdb():
    exp_name = "finetune_mmimdb"
    datasets = ["mmimdb"]
    loss_names = _loss_names({"mmimdb": 1})
#     loss_names = _loss_names({"mmimdb": 1, "prompt": -0.5})
    batch_size = 256
    max_epoch = 20
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-2
    val_check_interval = 0.2
    weight_decay = 2e-2
#     optim_type = "adam"
    max_text_len = 1024

@ex.named_config
def task_finetune_irtr_coco():
    exp_name = "finetune_irtr_coco"
    datasets = ["coco"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_coco_randaug():
    exp_name = "finetune_irtr_coco_randaug"
    datasets = ["coco"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_f30k():
    exp_name = "finetune_irtr_f30k"
    datasets = ["f30k"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_f30k_randaug():
    exp_name = "finetune_irtr_f30k_randaug"
    datasets = ["f30k"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def step25k():
    max_epoch = 100


@ex.named_config
def step50k():
    max_epoch = 100


@ex.named_config
def step100k():
    max_epoch = 100


@ex.named_config
def step200k():
    max_epoch = 200

@ex.named_config
def vit32_base():
    vit = "vit_base_patch32_384"
    patch_size = 32
    hidden_size = 768
    num_heads = 12
    num_layers = 12
