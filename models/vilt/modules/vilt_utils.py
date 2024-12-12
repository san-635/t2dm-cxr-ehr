from math import e
import torch
import random

from transformers.optimization import AdamW
from transformers import (get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup)

from vilt.modules.objectives import compute_irtr_recall
from vilt.modules.my_metrics import Accuracy, VQAScore, Scalar, F1_Score, AUROC, AUPRC, Scalar2, check
from vilt.modules.my_metrics import compute_ci

def set_metrics(pl_module):
    for split in ["train", "val"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v < 1:
                continue
            if k == "vqa":
                setattr(pl_module, f"{split}_vqa_score", VQAScore())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())

            elif k == "mmimdb":
                setattr(pl_module, f"{split}_{k}_F1_scores", F1_Score())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())

            elif k == "ehr_ecg_cxr":    # relevant for this study
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_AUROC", AUROC())
                setattr(pl_module, f"{split}_{k}_AUPRC", AUPRC())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())

            elif k == "hatememes":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_AUROC", AUROC())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                
            elif k == "food101":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())       
                
            elif k == "nlvr2":
                if split == "train":
                    setattr(pl_module, f"train_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"train_{k}_loss", Scalar())
                else:
                    setattr(pl_module, f"dev_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"dev_{k}_loss", Scalar())
                    setattr(pl_module, f"test_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"test_{k}_loss", Scalar())
            elif k == "irtr":
                setattr(pl_module, f"{split}_irtr_loss", Scalar())
            elif k == "mppd" or k == "mpfr":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "itm":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_wpa_loss", Scalar())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())

def test_ablation(pl_module, loss_name, res):
    test_ratio = pl_module.hparams.config['test_ratio']
    exp_name = pl_module.hparams.config["test_exp_name"]
    test_type = pl_module.hparams.config["test_type"]       
    records = f'missing ratio: {test_ratio}, ' + res
    record_file = f'./records/{loss_name}/{loss_name}_{exp_name}_on_missing_{test_type}'
    with open(record_file, 'a+') as f:
        f.write(records+'\n')
                
def epoch_wrapup(pl_module):
    phase = "train" if pl_module.training else "val"
    # the_metric = 0

    if pl_module.hparams.config["get_recall_metric"] and not pl_module.training:
        (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_irtr_recall(pl_module)
        print((ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10), pl_module.global_step)
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r1", ir_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r5", ir_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r10", ir_r10, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r1", tr_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r5", tr_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r10", tr_r10, pl_module.global_step
        )
        # the_metric += ir_r1.item() + tr_r1.item()

    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v < 1:
            continue

        value = 0

        if loss_name == "vqa":
            value = getattr(pl_module, f"{phase}_{loss_name}_score").compute()
            pl_module.log(f"{loss_name}/{phase}/score_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_score").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            
        # relevant for this study
        elif loss_name == "ehr_ecg_cxr":
            value1, logits1, targets1 = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()      # compute overall accuracy over all batches
            if pl_module.hparams.config["test_only"] == True:
                pl_module.log(f"{loss_name}/test/accuracy", value1)
                value1_upper, value1_lower = compute_ci(getattr(pl_module, f"{phase}_{loss_name}_accuracy"), logits1, targets1, value1, num_iter=1000)      # compute 95% CI for accuracy
                pl_module.log(f"{loss_name}/test/accuracy_CI_lower", value1_lower)
                pl_module.log(f"{loss_name}/test/accuracy_CI_upper", value1_upper)
            else:
                pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value1)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()

            value2, logits2, targets2 = getattr(pl_module, f"{phase}_{loss_name}_AUROC").compute()         # compute macro-AUROC over all batches
            if pl_module.hparams.config["test_only"] == True:
                pl_module.log(f"{loss_name}/test/AUROC", value2)
                value2_upper, value2_lower = compute_ci(getattr(pl_module, f"{phase}_{loss_name}_AUROC"), logits2, targets2, value2, num_iter=1000)      # compute 95% CI for AUROC
                pl_module.log(f"{loss_name}/test/AUROC_CI_lower", value2_lower)
                pl_module.log(f"{loss_name}/test/AUROC_CI_upper", value2_upper)
            else:
                pl_module.log(f"{loss_name}/{phase}/AUROC_epoch", value2)
            getattr(pl_module, f"{phase}_{loss_name}_AUROC").reset()

            value3, logits3, targets3 = getattr(pl_module, f"{phase}_{loss_name}_AUPRC").compute()         # compute AUPRC/average precision over all batches
            if pl_module.hparams.config["test_only"] == True:
                pl_module.log(f"{loss_name}/test/AUPRC", value3)
                value3_upper, value3_lower = compute_ci(getattr(pl_module, f"{phase}_{loss_name}_AUPRC"), logits3, targets3, value3, num_iter=1000)      # compute 95% CI for AUPRC
                pl_module.log(f"{loss_name}/test/AUPRC_CI_lower", value3_lower)
                pl_module.log(f"{loss_name}/test/AUPRC_CI_upper", value3_upper)
            else:
                pl_module.log(f"{loss_name}/{phase}/AUPRC_epoch", value3)
            getattr(pl_module, f"{phase}_{loss_name}_AUPRC").reset()

            loss_value = getattr(pl_module, f"{phase}_{loss_name}_loss").compute()                         # compute overall loss over all batches
            if pl_module.hparams.config["test_only"] == True:
                pl_module.log(f"{loss_name}/test/loss", loss_value)
            else:
                pl_module.log(f"{loss_name}/{phase}/loss_epoch", loss_value)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

            if pl_module.hparams.config["test_only"] == True:
                with open(pl_module.hparams.config["log_dir"]+'/test.txt', 'w') as file:
                    file.write(f'TEST RESULTS:\nAUPRC = {value3:4f} ({value3_lower:4f}, {value3_upper:4f})\nAUROC = {value2:4f} ({value2_lower:4f}, {value2_upper:4f})\nAccuracy =  {value1:4f} ({value1_lower:4f}, {value1_upper:4f})\nLoss = {loss_value:4f}')

            torch.cuda.empty_cache()

        elif loss_name == "hatememes":
            value2 = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value2)       
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            value = getattr(pl_module, f"{phase}_{loss_name}_AUROC").compute()
            pl_module.log(f"{loss_name}/{phase}/AUROC_epoch", value)            
            getattr(pl_module, f"{phase}_{loss_name}_AUROC").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()      
            
            if pl_module.hparams.config["test_exp_name"] is not None:
                res = 'AUROC: {0:.2f}, Accuracy: {1:.2f}'.format(100*value, 100*value2)
                test_ablation(pl_module, loss_name, res)
            
        elif loss_name == "food101":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)       
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()

            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()   
            
            if pl_module.hparams.config["test_exp_name"] is not None:
                res = 'Accuracy: {0:.2f}'.format(100*value)
                test_ablation(pl_module, loss_name, res)            
            
        elif loss_name == "mmimdb":
            values = getattr(pl_module, f"{phase}_{loss_name}_F1_scores").compute()
            value = values[1]
            pl_module.log(f"{loss_name}/{phase}/F1_Micro_epoch", values[0])
            pl_module.log(f"{loss_name}/{phase}/F1_Macro_epoch", values[1])
            pl_module.log(f"{loss_name}/{phase}/F1_Samples_epoch", values[2])
            pl_module.log(f"{loss_name}/{phase}/F1_Weighted_epoch", values[3])
            getattr(pl_module, f"{phase}_{loss_name}_F1_scores").reset()            
            
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            
            if pl_module.hparams.config["test_exp_name"] is not None:
                res = 'F1-Macro: {0:.2f}, F1-Micro: {1:.2f}, F1-Weighted: {2:.2f}, F1-Sample: {3:.2f}'.format(100*values[1], 100*values[0], 100*values[2], 100*values[3])
                test_ablation(pl_module, loss_name, res)              
            
        elif loss_name == "nlvr2":
            if phase == "train":
                value = getattr(pl_module, f"train_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/train/accuracy_epoch", value)
                getattr(pl_module, f"train_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/train/loss_epoch",
                    getattr(pl_module, f"train_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"train_{loss_name}_loss").reset()
            else:
                value = getattr(pl_module, f"dev_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/dev/accuracy_epoch", value)
                getattr(pl_module, f"dev_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/dev/loss_epoch",
                    getattr(pl_module, f"dev_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"dev_{loss_name}_loss").reset()

                value = getattr(pl_module, f"test_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/test/accuracy_epoch", value)
                getattr(pl_module, f"test_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/test/loss_epoch",
                    getattr(pl_module, f"test_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"test_{loss_name}_loss").reset()
        elif loss_name == "irtr":
            pl_module.log(
                f"{loss_name}/{phase}/irtr_loss_epoch",
                getattr(pl_module, f"{phase}_irtr_loss").compute(),
            )
            getattr(pl_module, f"{phase}_irtr_loss").reset()
        elif loss_name == "mppd" or loss_name == "mpfr":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "itm":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            pl_module.log(
                f"{loss_name}/{phase}/wpa_loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_wpa_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_wpa_loss").reset()
        else:
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        # the_metric += value


def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v >= 1
    ]
    return


def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]
    
    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["vqa_classifier", "ehr_ecg_cxr_classifier", "mmimdb_classifier", "food101_classifier", "hatememes_classifier", "nlvr2_classifier"]
    lr_mult = pl_module.hparams.config["lr_mult"]           # for classifier head layers
    end_lr = pl_module.hparams.config["end_lr"]             # lr decay end value
    decay_power = pl_module.hparams.config["decay_power"]   # 1 (linear) or cosine
    optim_type = pl_module.hparams.config["optim_type"]     # AdamW or Adam or SGD with momentum

    optimizer_grouped_parameters = [
        {
            # 1: Parameters with decay and not in classifier heads
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr,
        },            
        {
            # 2: Parameters without decay and not in classifier heads
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            # 3: Parameters with decay and in classifier heads
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            # 4: Parameters without decay and in classifier heads
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]
    
    # create optimizer
    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
    
    # determine max_steps and warmup_steps
    max_steps = (
        len(pl_module.trainer.datamodule.train_dataloader())
        * pl_module.trainer.max_epochs
        // pl_module.trainer.accumulate_grad_batches
    )

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)
    
    # create learning rate scheduler
    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:   # decay_power == 1 / linear
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )
