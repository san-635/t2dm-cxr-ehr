from __future__ import absolute_import
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fusions.lstm_module import LSTM
from fusions.resnet_module import ResNet
from fusions.fusion import EHR_ECG_CXR_Fusion, EHR_CXR_Fusion
from fusions.trainer import Trainer

class EHR_ECG_CXR_FusionTrainer(Trainer):
    def __init__(self, args, train_dl, val_dl, test_dl=None):
        super(EHR_ECG_CXR_FusionTrainer, self).__init__(args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args

        # DataLoaders
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

        # Model architecture
        self.ehr_model = LSTM(self.args, input_dim=self.args.ehr_n_var).to(self.device)
        self.ecg_model = LSTM(self.args, input_dim=self.args.ecg_n_var).to(self.device)
        self.cxr_model = ResNet(self.args, str(self.device)).to(self.device)
        self.model = EHR_ECG_CXR_Fusion(args, self.ehr_model, self.ecg_model, self.cxr_model).to(self.device)
        self.init_fusion_method()

        # Loss, Optimizer and LR Sceduler
        self.loss = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), args.lr, betas=(0.9, self.args.beta_1))
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')
        self.load_state()       # no operation if self.args.load_state is None

        self.epoch = 0
        self.val_epoch = 0
        self.patience = 0
        self.best_auroc = 0
        self.best_stats = None
        self.epochs_stats = {'Train loss': [], 'Val loss': [], 'Val AUROC': []}
    
    def init_fusion_method(self):
        # for stage == finetune_early and mode == train, load the three pretrained encoders
        if self.args.load_state_ehr is not None:
            self.load_ehr_pheno(load_state=self.args.load_state_ehr)
        if self.args.load_state_ecg is not None:
            self.load_ecg_pheno(load_state=self.args.load_state_ecg)
        if self.args.load_state_cxr is not None:
            self.load_cxr_pheno(load_state=self.args.load_state_cxr)
        # for mode == eval, load finetuned/trained model
        if self.args.load_state is not None:
            self.load_state()

        if self.args.stage == 'pretrain_ehr':
            self.freeze(self.model.ecg_model)
            self.freeze(self.model.cxr_model)
        elif self.args.stage == 'pretrain_ecg':
            self.freeze(self.model.ehr_model)
            self.freeze(self.model.cxr_model)
        elif self.args.stage == 'pretrain_cxr':
            self.freeze(self.model.ehr_model)
            self.freeze(self.model.ecg_model)
        elif self.args.stage == 'finetune_early':
            self.freeze(self.model.ehr_model)
            self.freeze(self.model.ecg_model)
            self.freeze(self.model.cxr_model)
        # nothing is frozen for finetune_joint

    def train_epoch(self):
        print(f'\nStarting training epoch {self.epoch}')
        epoch_loss = 0
        train_PRED = torch.FloatTensor().to(self.device)
        train_LABEL = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)

        for i, batch in enumerate(self.train_dl, start=1):
            # step / batch-level
            ehr = batch["text"].float().to(self.device)
            ehr_len = batch["t_len"]
            ecg = batch["ecg"].float().to(self.device)
            ecg_len = batch["e_len"]
            img = batch["image"][0].to(self.device)
            train_label = torch.tensor(batch["label"]).float().to(self.device)

            output = self.model(ehr, ehr_len, ecg, ecg_len, img)
            if output["preds"].shape[0] == 1:
                train_pred = output["preds"][-1]
            else:
                train_pred = output["preds"].squeeze()
            loss = self.loss(train_pred, train_label)
            epoch_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_PRED = torch.cat((train_PRED, train_pred), 0)
            train_LABEL = torch.cat((train_LABEL, train_label), 0)

            if i == 1 or i % 5 == 0:
                print(f"Training epoch: [{self.epoch} / {self.args.max_epoch}], step/batch: [{i} / {steps}], lr: {self.optimizer.param_groups[0]['lr']:0.4E}, loss so far: {epoch_loss/i:0.4f}")
            
            # validate model at mid and end of each training epoch
            if (i==steps//2) or (i==steps):
                self.model.eval()
                if i==steps//2:
                    self.val_epoch = 2*self.epoch - 1
                else:
                    self.val_epoch = 2*self.epoch
                val_ret = self.validate_epoch(self.val_dl)
                self.save_checkpoint(prefix='last')

                if self.best_auroc < val_ret['auroc_mean']:         # note: auroc_mean is same as auc_scores for binary classification
                    self.best_auroc = val_ret['auroc_mean']
                    self.best_stats = val_ret
                    self.save_checkpoint(prefix='best')
                    self.print_and_write_stats(val_ret, prefix='Val', isbest=True)
                    self.patience = 0
                else:
                    self.print_and_write_stats(val_ret, prefix='Val', isbest=False)
                    self.patience += 1
                self.model.train()
        
        # epoch-level
        ret = self.compute_metrics(train_LABEL.data.cpu().numpy(), train_PRED.data.cpu().numpy())
        self.epochs_stats['Train loss'].append(epoch_loss/i)
        return ret
    
    def validate_epoch(self, val_dl):
        print(f'\n\tStarting val epoch {self.val_epoch}')
        epoch_loss = 0
        val_PRED = torch.FloatTensor().to(self.device)
        val_LABEL = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            for i, batch in enumerate(val_dl, start=1):
                # step / batch-level
                ehr = Variable(batch["text"].float().to(self.device), requires_grad=False)
                ehr_len = batch["t_len"]
                ecg = Variable(batch["ecg"].float().to(self.device), requires_grad=False)
                ecg_len = batch["e_len"]
                img = batch["image"][0].to(self.device)
                val_label = Variable(torch.tensor(batch["label"]).float().to(self.device), requires_grad=False)

                output = self.model(ehr, ehr_len, ecg, ecg_len, img)
                if output["preds"].shape[0] == 1:
                    val_pred = output["preds"][-1]
                else:
                    val_pred = output["preds"].squeeze()
                loss = self.loss(val_pred, val_label)
                epoch_loss += loss.item()

                val_PRED = torch.cat((val_PRED, val_pred), 0)
                val_LABEL = torch.cat((val_LABEL, val_label), 0)
        
        # epoch-level
        print(f"\tVal epoch: [{self.val_epoch} / {(2*self.args.max_epoch)}], overall loss: {epoch_loss/i:0.4f}")
        self.scheduler.step(epoch_loss/len(self.val_dl))
        
        ret = self.compute_metrics(val_LABEL.data.cpu().numpy(), val_PRED.data.cpu().numpy())
        self.epochs_stats['Val loss'].append(epoch_loss/i)
        self.epochs_stats['Val AUROC'].append(ret['auroc_mean'])

        return ret
    
    def test_epoch(self, test_dl):
        test_PRED = torch.FloatTensor().to(self.device)
        test_LABEL = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            for batch in test_dl:
                # step / batch-level
                ehr = Variable(batch["text"].float().to(self.device), requires_grad=False)
                ehr_len = batch["t_len"]
                ecg = Variable(batch["ecg"].float().to(self.device), requires_grad=False)
                ecg_len = batch["e_len"]
                img = batch["image"][0].to(self.device)
                test_label = Variable(torch.tensor(batch["label"]).float().to(self.device), requires_grad=False)

                output = self.model(ehr, ehr_len, ecg, ecg_len, img)
                if output["preds"].shape[0] == 1:
                    test_pred = output["preds"][-1]
                else:
                    test_pred = output["preds"].squeeze()

                test_PRED = torch.cat((test_PRED, test_pred), 0)
                test_LABEL = torch.cat((test_LABEL, test_label), 0)
        
        # epoch-level
        ret = self.compute_metrics(test_LABEL.data.cpu().numpy(), test_PRED.data.cpu().numpy())
        return ret

    def train(self):
        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            self.model.train()
            self.train_epoch()
            self.plot_stats(self.epochs_stats, key='loss', filename='train_and_val_loss_plots.pdf')
            self.plot_stats(self.epochs_stats, key='AUROC', filename='val_auroc_plot.pdf')
            if self.patience >= self.args.patience:                 # early stopping
                break
        self.print_and_write_stats(self.best_stats, isbest=True)
    
    def eval(self):
        self.epoch = 0
        self.val_epoch = 0

        self.model.eval()
        ret = self.test_epoch(self.test_dl)
        self.print_and_write_stats(ret, isbest=True, prefix='Test', filename='results_test.txt')

class EHR_CXR_FusionTrainer(Trainer):
    def __init__(self, args, train_dl, val_dl, test_dl=None):
        super(EHR_CXR_FusionTrainer, self).__init__(args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args

        # DataLoaders
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

        # Model architecture
        self.ehr_model = LSTM(self.args, input_dim=self.args.ehr_n_var).to(self.device)
        self.cxr_model = ResNet(self.args, str(self.device)).to(self.device)
        self.model = EHR_CXR_Fusion(args, self.ehr_model, self.cxr_model).to(self.device)
        self.init_fusion_method()

        # Loss, Optimizer and LR Sceduler
        self.loss = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), args.lr, betas=(0.9, self.args.beta_1))
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')
        self.load_state()       # no operation if self.args.load_state is None

        self.epoch = 0
        self.val_epoch = 0
        self.patience = 0
        self.best_auroc = 0
        self.best_stats = None
        self.epochs_stats = {'Train loss': [], 'Val loss': [], 'Val AUROC': []}
    
    def init_fusion_method(self):
        # for stage == finetune_early and mode == train, load both pretrained encoders
        if self.args.load_state_ehr is not None:
            self.load_ehr_pheno(load_state=self.args.load_state_ehr)
        if self.args.load_state_cxr is not None:
            self.load_cxr_pheno(load_state=self.args.load_state_cxr)
        # for mode == eval, load finetuned/trained model
        if self.args.load_state is not None:
            self.load_state()

        if self.args.stage == 'pretrain_ehr':
            self.freeze(self.model.cxr_model)
        elif self.args.stage == 'pretrain_cxr':
            self.freeze(self.model.ehr_model)
        elif self.args.stage == 'finetune_early':
            self.freeze(self.model.ehr_model)
            self.freeze(self.model.cxr_model)
        # nothing is frozen for finetune_joint

    def train_epoch(self):
        print(f'\nStarting training epoch {self.epoch}')
        epoch_loss = 0
        train_PRED = torch.FloatTensor().to(self.device)
        train_LABEL = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)

        for i, batch in enumerate(self.train_dl, start=1):
            # step / batch-level
            ehr = batch["text"].float().to(self.device)
            ehr_len = batch["t_len"]
            img = batch["image"][0].to(self.device)
            train_label = torch.tensor(batch["label"]).float().to(self.device)

            output = self.model(ehr, ehr_len, img)
            if output["preds"].shape[0] == 1:
                train_pred = output["preds"][-1]
            else:
                train_pred = output["preds"].squeeze()
            loss = self.loss(train_pred, train_label)
            epoch_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_PRED = torch.cat((train_PRED, train_pred), 0)
            train_LABEL = torch.cat((train_LABEL, train_label), 0)

            if i == 1 or i % 5 == 0:
                print(f"Training epoch: [{self.epoch} / {self.args.max_epoch}], step/batch: [{i} / {steps}], lr: {self.optimizer.param_groups[0]['lr']:0.4E}, loss so far: {epoch_loss/i:0.4f}")
            
            # validate model at mid and end of each training epoch
            if (i==steps//2) or (i==steps):
                self.model.eval()
                if i==steps//2:
                    self.val_epoch = 2*self.epoch - 1
                else:
                    self.val_epoch = 2*self.epoch
                val_ret = self.validate_epoch(self.val_dl)
                self.save_checkpoint(prefix='last')

                if self.best_auroc < val_ret['auroc_mean']:         # note: auroc_mean is same as auc_scores for binary classification
                    self.best_auroc = val_ret['auroc_mean']
                    self.best_stats = val_ret
                    self.save_checkpoint(prefix='best')
                    self.print_and_write_stats(val_ret, prefix='Val', isbest=True)
                    self.patience = 0
                else:
                    self.print_and_write_stats(val_ret, prefix='Val', isbest=False)
                    self.patience += 1
                self.model.train()
        
        # epoch-level
        ret = self.compute_metrics(train_LABEL.data.cpu().numpy(), train_PRED.data.cpu().numpy())
        self.epochs_stats['Train loss'].append(epoch_loss/i)
        return ret
    
    def validate_epoch(self, val_dl):
        print(f'\n\tStarting val epoch {self.val_epoch}')
        epoch_loss = 0
        val_PRED = torch.FloatTensor().to(self.device)
        val_LABEL = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            for i, batch in enumerate(val_dl, start=1):
                # step / batch-level
                ehr = Variable(batch["text"].float().to(self.device), requires_grad=False)
                ehr_len = batch["t_len"]
                img = batch["image"][0].to(self.device)
                val_label = Variable(torch.tensor(batch["label"]).float().to(self.device), requires_grad=False)

                output = self.model(ehr, ehr_len, img)
                if output["preds"].shape[0] == 1:
                    val_pred = output["preds"][-1]
                else:
                    val_pred = output["preds"].squeeze()
                loss = self.loss(val_pred, val_label)
                epoch_loss += loss.item()

                val_PRED = torch.cat((val_PRED, val_pred), 0)
                val_LABEL = torch.cat((val_LABEL, val_label), 0)
        
        # epoch-level
        print(f"\tVal epoch: [{self.val_epoch} / {(2*self.args.max_epoch)}], overall loss: {epoch_loss/i:0.4f}")
        self.scheduler.step(epoch_loss/len(self.val_dl))
        
        ret = self.compute_metrics(val_LABEL.data.cpu().numpy(), val_PRED.data.cpu().numpy())
        self.epochs_stats['Val loss'].append(epoch_loss/i)
        self.epochs_stats['Val AUROC'].append(ret['auroc_mean'])

        return ret
    
    def test_epoch(self, test_dl):
        test_PRED = torch.FloatTensor().to(self.device)
        test_LABEL = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            for batch in test_dl:
                # step / batch-level
                ehr = Variable(batch["text"].float().to(self.device), requires_grad=False)
                ehr_len = batch["t_len"]
                img = batch["image"][0].to(self.device)
                test_label = Variable(torch.tensor(batch["label"]).float().to(self.device), requires_grad=False)

                output = self.model(ehr, ehr_len, img)
                if output["preds"].shape[0] == 1:
                    test_pred = output["preds"][-1]
                else:
                    test_pred = output["preds"].squeeze()

                test_PRED = torch.cat((test_PRED, test_pred), 0)
                test_LABEL = torch.cat((test_LABEL, test_label), 0)
        
        # epoch-level
        ret = self.compute_metrics(test_LABEL.data.cpu().numpy(), test_PRED.data.cpu().numpy())
        return ret

    def train(self):
        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            self.model.train()
            self.train_epoch()
            self.plot_stats(self.epochs_stats, key='loss', filename='train_and_val_loss_plots.pdf')
            self.plot_stats(self.epochs_stats, key='AUROC', filename='val_auroc_plot.pdf')
            if self.patience >= self.args.patience:                 # early stopping
                break
        self.print_and_write_stats(self.best_stats, isbest=True)
    
    def eval(self):
        self.epoch = 0
        self.val_epoch = 0

        self.model.eval()
        ret = self.test_epoch(self.test_dl)
        self.print_and_write_stats(ret, isbest=True, prefix='Test', filename='results_test.txt')