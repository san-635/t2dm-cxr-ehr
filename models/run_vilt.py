import warnings
import os
import copy
import pytorch_lightning as pl
import numpy as np
import torch

from vilt.config import ex
from vilt.modules.vilt_model import ViLT
from vilt.datamodule import EHR_ECG_CXR_DataModule

warnings.filterwarnings("ignore")

@ex.automain
def main(_config):
    # Set seeds for reproducibility #
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    torch.manual_seed(_config["seed"])
    np.random.seed(_config["seed"])
    
    # Initialise data loader, model and logger #
    dm = EHR_ECG_CXR_DataModule(_config)
    model = ViLT(_config)
    os.makedirs(_config["log_dir"], exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(_config["log_dir"])
    
    # Callbacks #
    # Save best model based on val AUROC
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=_config["exp_name"].split("finetune_")[-1]+"/val/AUROC_epoch",
        mode="max",
        save_top_k=1,
        save_last=False,
        verbose=True
    )

    # Early stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor=_config["exp_name"].split("finetune_")[-1]+"/val/AUROC_epoch",
        mode="max",
        min_delta=0,
        patience=2//_config["val_check_interval"],
        verbose=True
    )
    
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback, early_stop_callback]
    
    # GPU settings #
    # Uncomment the following lines if you want to limit the GPU memory usage
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(_config["gpu_device"])
    # torch.cuda.set_per_process_memory_fraction(0.5, device=0)

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (_config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"])
    print(_config["batch_size"], _config["per_gpu_batchsize"], num_gpus, _config["num_nodes"])

    # Initialise trainer #
    trainer = pl.Trainer(
        gpus=str(_config["gpu_device"]),
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="cuda",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"],
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    # Train or test model #
    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)