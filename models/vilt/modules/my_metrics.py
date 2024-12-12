import numpy as np
import pandas as pd
from sympy import true
import torch
from torchmetrics.functional.classification import binary_f1_score, multiclass_f1_score, binary_auroc, multiclass_auroc
from torchmetrics.functional.classification import binary_average_precision, multiclass_average_precision
from pytorch_lightning.metrics.metric import Metric

# relevant for this study (accuracy)
class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")        

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )

        self.logits.append(logits)
        self.targets.append(target)

    def compute(self):
        if type(self.logits) == list:
            all_logits = torch.cat(self.logits)
            all_targets = torch.cat(self.targets).long()
        else:
            all_logits = self.logits
            all_targets = self.targets.long()
        
        if all_logits.size(-1)>1:                               # multi-class classification
            preds = all_logits.argmax(dim=-1)
        else:                                                   # binary classification
            all_logits.squeeze_(-1)
            preds = (torch.sigmoid(all_logits)>0.5).long()
            
        if all_targets.numel() == 0:                            # avoid division by zero
            return 1

        assert preds.shape == all_targets.shape
        acc = torch.sum(preds == all_targets) / all_targets.numel()
        return acc, self.logits, self.targets                   # overall accuracy across all samples

# relevant for this study (AUROC)
class AUROC(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, logits, target):
        logits, targets = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        
        self.logits.append(logits)
        self.targets.append(targets)

    def compute(self):
        if type(self.logits) == list:
            all_logits = torch.cat(self.logits)
            all_targets = torch.cat(self.targets).long()
        else:
            all_logits = self.logits
            all_targets = self.targets.long()

        if all_logits.size(-1)>1:                               # multi-class classification
            all_logits = torch.softmax(all_logits.items(), dim=1)
            AUROC = multiclass_auroc(all_logits, all_targets, average=None, num_classes=all_logits.size(-1))[1]
        else:                                                   # binary classification
            preds = torch.sigmoid(all_logits.squeeze_(-1))
            AUROC = binary_auroc(preds, all_targets)
        return AUROC, self.logits, self.targets                 # overall macro-AUROC across all samples

# relevant for this study (AUPRC)
class AUPRC(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, logits, target):
        logits, targets = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        
        self.logits.append(logits)
        self.targets.append(targets)

    def compute(self):
        if type(self.logits) == list:
            all_logits = torch.cat(self.logits)
            all_targets = torch.cat(self.targets).long()
        else:
            all_logits = self.logits
            all_targets = self.targets.long()

        if all_logits.size(-1)>1:                               # multi-class classification
            all_logits = torch.softmax(all_logits.items(), dim=1)
            AUPRC = multiclass_average_precision(all_logits, all_targets, average=None, num_classes=all_logits.size(-1))[1]
        else:                                                   # binary classification
            preds = torch.sigmoid(all_logits.squeeze_(-1))
            AUPRC = binary_average_precision(preds, all_targets)
        return AUPRC, self.logits, self.targets                 # overall AUPRC across all samples

# relevant for this study (loss)
class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total

class F1_Score(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, logits, target):
        logits, targets = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        
        self.logits.append(logits)
        self.targets.append(targets)

    def compute(self, use_sigmoid=True):
        if type(self.logits) == list:
            all_logits = torch.cat(self.logits)
            all_targets = torch.cat(self.targets).long()
        else:
            all_logits = self.logits
            all_targets = self.targets.long()
        if all_logits.size(-1)>1:
            all_preds = all_logits.argmax(dim=-1)
            F1_score = multiclass_f1_score(all_preds, all_targets, average=None, num_classes=all_logits.size(-1))[1]
        else:
            all_preds = (torch.sigmoid(all_logits.squeeze_(-1))>0.5).long()
            F1_score = binary_f1_score(all_preds, all_targets)
        return (F1_score)
    
class check(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, logits, target):
        logits, targets = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        
        self.logits.append(logits)
        self.targets.append(targets)

    def compute(self, use_sigmoid=True):
        if type(self.logits) == list:
            all_logits = torch.cat(self.logits).long()
            all_targets = torch.cat(self.targets).long()
        else:
            all_logits = self.logits.long()
            all_targets = self.targets.long()

        mislead = all_logits ^ all_targets
        accuracy = mislead.sum(dim=0)
        return accuracy

class Scalar2(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar, num):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        
        self.scalar += scalar
        self.total += num

    def compute(self):
        return self.scalar / self.total

class VQAScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total
    
# relevant for this study (95% confidence interval for metrics)
def compute_ci(metric, logits, targets, true_value, num_iter=1000):
    logits = torch.cat(logits)
    targets = torch.cat(targets)
    df = pd.DataFrame({"preds": logits.squeeze_(-1).cpu(), "labels": targets.cpu()})        # moved to CPU for DF creation

    sample_values = []
    for _ in range(num_iter):
        sample = df.sample(frac=1, replace=True)
        sample_logits = torch.tensor(sample['preds'].values).to(logits.device)              # moved to GPU for sigmoid later
        sample_targets = torch.tensor(sample['labels'].values).long().to(targets.device)    # moved to GPU for later operations
        
        if isinstance(metric, Accuracy):
            preds = (torch.sigmoid(sample_logits)>0.5).long()
            sample_value = torch.sum(preds == sample_targets) / sample_targets.numel()
        elif isinstance(metric, AUROC):
            preds = torch.sigmoid(sample_logits)
            sample_value = binary_auroc(preds, sample_targets)
        elif isinstance(metric, AUPRC):
            preds = torch.sigmoid(sample_logits)
            sample_value = binary_average_precision(preds, sample_targets)
        
        sample_values.append(sample_value)                                                  # still on GPU

    delta = [(true_value - sample_value) for sample_value in sample_values]                 # still on GPU
    delta = [d.cpu() for d in delta]                                                        # moved to CPU for numpy operations
    delta = list(np.sort(delta))
    delta_lower = np.percentile(delta, 97.5)
    delta_upper = np.percentile(delta, 2.5)

    upper = true_value - delta_upper
    lower = true_value - delta_lower
    return upper, lower                                                                     # moved to GPU since true_value is on GPU