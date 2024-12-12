from __future__ import absolute_import
from __future__ import print_function
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt

from fusions.metrics import get_model_performance

class Trainer():
    def __init__(self, args):
        self.args = args
        self.time_start = time.time()
        self.time_end = time.time()
        self.start_epoch = 1
        self.patience = 0

    def train(self):
        pass        # defined in fusion_trainer.py

    def train_epoch(self):
        pass        # defined in fusion_trainer.py

    def validate_epoch(self):
        pass        # defined in fusion_trainer.py

    def load_ehr_pheno(self, load_state):
        checkpoint = torch.load(load_state)
        own_state = self.model.state_dict()

        for name, param in checkpoint['state_dict'].items():
            if name not in own_state or 'ehr_model' not in name:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            own_state[name].copy_(param)

        print(f'Loaded pre-trained EHR encoder from {load_state}')

    def load_ecg_pheno(self, load_state):
        checkpoint = torch.load(load_state)
        own_state = self.model.state_dict()

        for name, param in checkpoint['state_dict'].items():
            if name not in own_state or 'ecg_model' not in name:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            own_state[name].copy_(param)

        print(f'Loaded pre-trained ECG encoder from {load_state}')

    def load_cxr_pheno(self, load_state):
        checkpoint = torch.load(load_state)
        own_state = self.model.state_dict()

        for name, param in checkpoint['state_dict'].items():
            if name not in own_state or 'cxr_model' not in name:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            own_state[name].copy_(param)

        print(f'Loaded pre-trained CXR encoder from {load_state}')

    def load_state(self):
        if self.args.load_state is None:
            return
        checkpoint = torch.load(self.args.load_state)
        own_state = self.model.state_dict()

        for name, param in checkpoint['state_dict'].items():
            if name not in own_state:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            own_state[name].copy_(param)
        print(f'Loaded model from {self.args.load_state}')

    def freeze(self, model):
        for p in model.parameters():
           p.requires_grad = False
    
    def compute_metrics(self, y_true, predictions):
        y_true = np.array(y_true)
        predictions = np.array(predictions)

        auc_scores = []
        auprc_scores = []
        acc_scores = []
        ci_auroc = []
        ci_auprc = []
        ci_acc = []
        if len(y_true.shape) == 1:              # for binary classification, convert to (n,1)
            y_true = y_true[:, None]
            predictions = predictions[:, None]
        for i in range(y_true.shape[1]):
            df = pd.DataFrame({'y_truth': y_true[:, i], 'y_pred': predictions[:, i]})
            (test_auprc, upper_auprc, lower_auprc), (test_auroc, upper_auroc, lower_auroc), (test_accuracy, upper_accuracy, lower_accuracy) = get_model_performance(df)
            auc_scores.append(test_auroc)
            auprc_scores.append(test_auprc)
            acc_scores.append(test_accuracy)
            ci_auroc.append((lower_auroc, upper_auroc))
            ci_auprc.append((lower_auprc, upper_auprc))
            ci_acc.append((lower_accuracy, upper_accuracy))
        
        auc_scores = np.array(auc_scores)
        auprc_scores = np.array(auprc_scores)
        acc_scores = np.array(acc_scores)
        ret_metrics = {
            "auc_scores": auc_scores,
            "auprc_scores": auprc_scores,
            "accuracy_scores": acc_scores,
            "auroc_mean": np.mean(auc_scores),
            "auprc_mean": np.mean(auprc_scores),
            "accuracy_mean": np.mean(acc_scores),
            "ci_auroc": ci_auroc,
            "ci_auprc": ci_auprc,
            "ci_accuracy": ci_acc
            }
        return ret_metrics

    def get_eta(self, epoch, iter):
        done_epoch = epoch - self.start_epoch
        remaining_epochs = self.args.max_epoch - epoch

        iter +=1
        self.time_end = time.time()
        delta = self.time_end - self.time_start
        
        done_iters = len(self.train_dl) * done_epoch + iter    
        remaining_iters = len(self.train_dl) * remaining_epochs - iter

        delta = (delta/done_iters)*remaining_iters
        
        sec = timedelta(seconds=int(delta))
        d = (datetime(1,1,1) + sec)
        eta = f"{d.day-1} Days {d.hour}:{d.minute}:{d.second}"
        return eta

    def plot_stats(self, ret, key='loss', filename='training_stats.pdf'):
        for loss in ret:
            if key in loss:
                plt.plot(ret[loss], label = f"{loss}")
        plt.xlabel('epochs')
        plt.ylabel(key)
        plt.title(key)
        plt.legend()
        plt.savefig(f"{self.args.save_dir}/{filename}")
        plt.close()
    
    def save_checkpoint(self, prefix='best'):
        path = f'{self.args.save_dir}/{prefix}_checkpoint.pth.tar'
        ret = {
            "epoch": self.val_epoch,
            "state_dict": self.model.state_dict(),
            "best_auroc": self.best_auroc,
            "optimizer": self.optimizer.state_dict(),
            "epochs_stats": self.epochs_stats
            }
        torch.save(ret, path)
        print(f"\tSaving {prefix} model at val epoch {self.val_epoch}")

    def print_and_write_stats(self, ret, prefix='Val', isbest=False, filename='results.txt'):
        with open(f"{self.args.save_dir}/{filename}", 'a') as results_file:
            if isbest:
                ci_auroc_all = []
                ci_auprc_all = []
                ci_acc_all = []

                if len(ret['auc_scores'].shape) > 1:    # for multiclass classification
                    for index, class_auc in enumerate(ret['auc_scores']):
                        line = f'{self.val_dl.dataset.CLASSES[index]:<90} & {class_auc:0.4f} ({ret["ci_auroc"][index][1]:0.4f}, {ret["ci_auroc"][index][0]:0.4f}) & {ret["auprc_scores"][index]:0.4f} ({ret["ci_auprc"][index][1]:0.4f}, {ret["ci_auprc"][index][0]:0.4f}) & {ret["accuracy_scores"][index]:0.4f} ({ret["ci_accuracy"][index][1]:0.4f}, {ret["ci_accuracy"][index][0]:0.4f})' 
                        ci_auroc_all.append([ret["ci_auroc"][index][0], ret["ci_auroc"][index][1]])
                        ci_auprc_all.append([ret["ci_auprc"][index][0], ret["ci_auprc"][index][1]])
                        ci_acc_all.append([ret["ci_accuracy"][index][0], ret["ci_accuracy"][index][1]])
                        print(line)
                        results_file.write(line)
                
                else:                                   # for binary classification
                    ci_auroc_all.append([ret["ci_auroc"][0][0], ret["ci_auroc"][0][1]])
                    ci_auprc_all.append([ret["ci_auprc"][0][0], ret["ci_auprc"][0][1]])
                    ci_acc_all.append([ret["ci_accuracy"][0][0], ret["ci_accuracy"][0][1]])

                ci_auroc_all = np.array(ci_auroc_all)
                ci_auprc_all = np.array(ci_auprc_all)
                ci_acc_all = np.array(ci_acc_all)

                line = f"{prefix} epoch {self.val_epoch:<3}\t best AUROC: {ret['auroc_mean']:0.4f} ({np.mean(ci_auroc_all[:, 0]):0.4f}, {np.mean(ci_auroc_all[:, 1]):0.4f}), best AUPRC: {ret['auprc_mean']:0.4f} ({np.mean(ci_auprc_all[:, 0]):0.4f}, {np.mean(ci_auprc_all[:, 1]):0.4f}), best accuracy: {ret['accuracy_mean']:0.4f} ({np.mean(ci_acc_all[:, 0]):0.4f}, {np.mean(ci_acc_all[:, 1]):0.4f})\n"
                print(f"\t{line}")
                results_file.write(line)
            else:
                line = f"{prefix} epoch {self.val_epoch:<3}\t last AUROC: {ret['auroc_mean']:0.4f}, last AUPRC: {ret['auprc_mean']:0.4f}, last accuracy: {ret['accuracy_mean']:0.4f}\n"
                print(f"\t{line}")
                results_file.write(line)