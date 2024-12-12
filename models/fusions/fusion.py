import torch.nn as nn
import torch

class EHR_ECG_CXR_Fusion(nn.Module):
    def __init__(self, args, ehr_model, ecg_model, cxr_model):
        super(EHR_ECG_CXR_Fusion, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.ecg_model = ecg_model
        self.cxr_model = cxr_model

        assert self.ehr_model.feats_dim == self.ecg_model.feats_dim
        lstm_in = self.ehr_model.feats_dim
        resnet_out = self.cxr_model.feats_dim
        self.projection = nn.Linear(resnet_out, lstm_in)    # project CXR feature repr. to EHR/ECG feature repr. space
        
        feats_dim = 3 * self.ehr_model.feats_dim            # EHR, ECG and CXR features are concatenated
        self.fused_classifier = nn.Sequential(nn.Linear(feats_dim, self.args.num_classes), nn.Sigmoid())

    def forward_ehr(self, ehr, ehr_len):
        ehr_preds, ehr_feats = self.ehr_model(ehr, ehr_len)
        ret = {
            'preds': ehr_preds,
            'ehr_feats': ehr_feats
        }
        return ret
    
    def forward_ecg(self, ecg, ecg_len):
        ecg_preds, ecg_feats = self.ecg_model(ecg, ecg_len)
        ret = {
            'preds': ecg_preds,
            'ecg_feats': ecg_feats
        }
        return ret
    
    def forward_cxr(self, img):
        cxr_preds, _ , cxr_feats = self.cxr_model(img)
        ret = {
            'preds': cxr_preds,
            'cxr_feats': cxr_feats
        }
        return ret
    
    def forward_fusion(self, ehr, ehr_len, ecg, ecg_len, img):
        _, ehr_feats = self.ehr_model(ehr, ehr_len)
        _, ecg_feats = self.ecg_model(ecg, ecg_len)
        _, _ , cxr_feats = self.cxr_model(img)
        proj_cxr_feats = self.projection(cxr_feats)
        feats = torch.cat([ehr_feats, ecg_feats, proj_cxr_feats], dim=1)
        fused_preds = self.fused_classifier(feats)
        ret = {
            'preds': fused_preds,
            'ehr_feats': ehr_feats,
            'ecg_feats': ecg_feats,
            'cxr_feats': proj_cxr_feats
        }
        return ret
    
    def forward(self, ehr, ehr_len, ecg, ecg_len, img):
        if self.args.stage == 'pretrain_ehr':
            return self.forward_ehr(ehr, ehr_len)
        elif self.args.stage == 'pretrain_ecg':
            return self.forward_ecg(ecg, ecg_len)
        elif self.args.stage == 'pretrain_cxr':
            return self.forward_cxr(img)
        elif self.args.stage == 'finetune_early' or self.args.stage == 'finetune_joint':
            return self.forward_fusion(ehr, ehr_len, ecg, ecg_len, img)
        
class EHR_CXR_Fusion(nn.Module):
    def __init__(self, args, ehr_model, cxr_model):
        super(EHR_CXR_Fusion, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model

        lstm_in = self.ehr_model.feats_dim
        resnet_out = self.cxr_model.feats_dim
        self.projection = nn.Linear(resnet_out, lstm_in)    # project CXR feature repr. to EHR feature repr. space
        
        feats_dim = 2 * self.ehr_model.feats_dim            # EHR and CXR features are concatenated
        self.fused_classifier = nn.Sequential(nn.Linear(feats_dim, self.args.num_classes), nn.Sigmoid())

    def forward_ehr(self, ehr, ehr_len):
        ehr_preds, ehr_feats = self.ehr_model(ehr, ehr_len)
        ret = {
            'preds': ehr_preds,
            'ehr_feats': ehr_feats
        }
        return ret
    
    def forward_cxr(self, img):
        cxr_preds, _ , cxr_feats = self.cxr_model(img)
        ret = {
            'preds': cxr_preds,
            'cxr_feats': cxr_feats
        }
        return ret
    
    def forward_fusion(self, ehr, ehr_len, img):
        _, ehr_feats = self.ehr_model(ehr, ehr_len)
        _, _ , cxr_feats = self.cxr_model(img)
        proj_cxr_feats = self.projection(cxr_feats)
        feats = torch.cat([ehr_feats, proj_cxr_feats], dim=1)
        fused_preds = self.fused_classifier(feats)
        ret = {
            'preds': fused_preds,
            'ehr_feats': ehr_feats,
            'cxr_feats': proj_cxr_feats
        }
        return ret
    
    def forward(self, ehr, ehr_len, img):
        if self.args.stage == 'pretrain_ehr':
            return self.forward_ehr(ehr, ehr_len)
        elif self.args.stage == 'pretrain_cxr':
            return self.forward_cxr(img)
        elif self.args.stage == 'finetune_early' or self.args.stage == 'finetune_joint':
            return self.forward_fusion(ehr, ehr_len, img)