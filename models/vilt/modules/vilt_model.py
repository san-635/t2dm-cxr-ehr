from cgitb import text
import torch
import torch.nn as nn
import pytorch_lightning as pl

import vilt.modules.vision_transformer as vit
from vilt.modules import heads, objectives, vilt_utils

class ViLT(pl.LightningModule):
    # ===================== Model initialisation ===================== #
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        # EHR embedding module
        self.text_embeddings = heads.EHREmbedding(config["ehr_n_var"], config["hidden_size"])
        self.text_embeddings.apply(objectives.init_weights)

        # ECG embedding module
        self.ecg_embeddings = heads.ECGEmbedding(config["hidden_size"])
        self.ecg_embeddings.apply(objectives.init_weights)

        # modality-type embedding module
        self.token_type_embeddings = nn.Embedding(3, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        # set up ViT model
        # self.transformer is an instance of VisionTransformer class from vision_transformer.py
        if self.hparams.config["load_path"] == "":  # load vit_base_patch32_384 with pretrained weighs if no load_path is provided
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:                                       # load vit_base_patch32_384 with random weighs if load_path is provided
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        # pooling layer
        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        # for training (i.e., finetuning), load pretrained ViLT model's weights
        if (self.hparams.config["load_path"] != "" and not self.hparams.config["test_only"]):
            if self.hparams.config["load_path"] == "No":
                self.apply(objectives.init_weights)                                     # train ViLT from scratch
            else:
                ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu") # load vilt_200k_mlm_itm.ckpt
                state_dict = ckpt["state_dict"]
                if config["max_text_len"] != 40:                    # replace position embeddings weights with interpolated ones if max_text_len != 40
                    state_dict['text_embeddings.position_ids'] = torch.Tensor(range(config["max_text_len"])).long().view(1,-1)
                    pos_emb = state_dict['text_embeddings.position_embeddings.weight']
                    pos_emb = torch.nn.functional.interpolate(pos_emb.view(1,1,40,768), size=(config["max_text_len"],768), mode='bilinear').squeeze()
                    state_dict['text_embeddings.position_embeddings.weight'] = pos_emb
                if self.token_type_embeddings.weight.shape[0] > 2:  # add extra rows of rand to token_type_embeddings if there are more than 2 input modalities
                    num_extra = self.token_type_embeddings.weight.shape[0] - 2
                    extra_rows = torch.randn((num_extra, 768))
                    state_dict['token_type_embeddings.weight'] = torch.cat([state_dict['token_type_embeddings.weight'], extra_rows], dim=0)
                self.load_state_dict(state_dict, strict=False)

        # binary classifier
        hs = self.hparams.config["hidden_size"]
        if self.hparams.config["loss_names"]["ehr_ecg_cxr"] > 0:
            self.ehr_ecg_cxr_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 1),
            )
            self.ehr_ecg_cxr_classifier.apply(objectives.init_weights)

        # define metrics for each task (accuracy, AUROC, AUPRC for this study)
        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # for testing, load finetuned ViLT model's weights
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

    # ===================== Forward pass ===================== #
    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
        is_train=None,
    ):
        if self.hparams.config["modalities"] == "EHR+ECG+CXR":
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"

            # image embeddings - pass image data through ViT's visual_embed module
            if image_embeds is None and image_masks is None:
                if self.hparams.config["ablation"] == "missing" and self.hparams.config["test_only"]:
                    img = batch[imgkey][0]
                    (
                        image_embeds,
                        image_masks,
                        _,
                        _,
                    ) = self.transformer.visual_embed_for_missing(
                        img,
                        max_image_len=self.hparams.config["max_image_len"],
                    )
                else:
                    img = batch[imgkey][0]
                    (
                        image_embeds,
                        image_masks,
                        patch_index,
                        image_labels,
                    ) = self.transformer.visual_embed(
                        img,
                        max_image_len=self.hparams.config["max_image_len"],
                        mask_it=mask_image,
                    )
            
            text_embeds = self.text_embeddings(batch["text"])
            ecg_embeds = self.ecg_embeddings(batch["ecg"])
            
            # add modality-type embeddings (tensors of 0s for text, 1s for image, 2s for ECG) to text, image and ECG embeddings
            text_embeds, image_embeds, ecg_embeds = (
                text_embeds + self.token_type_embeddings(torch.zeros_like(text_embeds[:, :, 0]).long()),
                image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx).long()),
                ecg_embeds + self.token_type_embeddings(torch.full_like(ecg_embeds[:, :, 0], 2).long())
            )

            # prepend cls_token to masks of EHR and ECG data
            text_masks = batch["t_mask"]
            text_cls_token = torch.zeros((text_masks.size(0), 1), device=text_masks.device)
            text_masks = torch.cat([text_cls_token, text_masks], dim=1)

            ecg_masks = batch["e_mask"]
            ecg_cls_token = torch.zeros((ecg_masks.size(0), 1), device=ecg_masks.device)
            ecg_masks = torch.cat([ecg_cls_token, ecg_masks], dim=1)

            # concatenate by columns all embeddings, as well as all masks
            co_embeds = torch.cat([text_embeds, image_embeds, ecg_embeds], dim=1)
            co_masks = torch.cat([text_masks, image_masks, ecg_masks], dim=1)

            # pass concatenated embeddings and masks through transformer encoder
            x = co_embeds
            for i, blk in enumerate(self.transformer.blocks):
                x, _attn = blk(x, mask=co_masks)

            # normalise the concatenated transformer output
            x = self.transformer.norm(x)

            # separate feature representations from transformer output
            text_feats, image_feats, ecg_feats = (x[:, : text_embeds.shape[1]], x[:, text_embeds.shape[1] : -ecg_embeds.shape[1]], x[:, -ecg_embeds.shape[1] :])

            # pass the cls_token from transformer output through pooling layer to get cls_feats
            cls_feats = self.pooler(x)

            ret = {
                "text_feats": text_feats,
                "image_feats": image_feats,
                "ecg_feats": ecg_feats,
                "cls_feats": cls_feats,
                "raw_cls_feats": x[:, 0],
                "image_masks": image_masks,
                "text_masks": text_masks,
                "ecg_masks": ecg_masks,
            }

        elif self.hparams.config["modalities"] == "EHR+CXR":
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"

            # image embeddings - pass image data through ViT's visual_embed module
            if image_embeds is None and image_masks is None:
                if self.hparams.config["ablation"] == "missing" and self.hparams.config["test_only"]:
                    img = batch[imgkey][0]
                    (
                        image_embeds,
                        image_masks,
                        _,
                        _,
                    ) = self.transformer.visual_embed_for_missing(
                        img,
                        max_image_len=self.hparams.config["max_image_len"],
                    )
                else:
                    img = batch[imgkey][0]
                    (
                        image_embeds,
                        image_masks,
                        patch_index,
                        image_labels,
                    ) = self.transformer.visual_embed(
                        img,
                        max_image_len=self.hparams.config["max_image_len"],
                        mask_it=mask_image,
                    )
            
            text_embeds = self.text_embeddings(batch["text"])
            
            # add modality-type embeddings (tensors of 0s for text, 1s for image) to text and image embeddings
            text_embeds, image_embeds = (
                text_embeds + self.token_type_embeddings(torch.zeros_like(text_embeds[:, :, 0]).long()),
                image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx).long())
            )

            # prepend cls_token to masks of EHR data
            text_masks = batch["t_mask"]
            text_cls_token = torch.zeros((text_masks.size(0), 1), device=text_masks.device)
            text_masks = torch.cat([text_cls_token, text_masks], dim=1)

            # concatenate by columns all embeddings, as well as all masks
            co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
            co_masks = torch.cat([text_masks, image_masks], dim=1)

            # pass concatenated embeddings and masks through transformer encoder
            x = co_embeds
            for i, blk in enumerate(self.transformer.blocks):
                x, _attn = blk(x, mask=co_masks)

            # normalise the concatenated transformer output
            x = self.transformer.norm(x)

            # separate feature representations from transformer output
            text_feats, image_feats = (x[:, : text_embeds.shape[1]], x[:, text_embeds.shape[1] :])

            # pass the cls_token from transformer output through pooling layer to get cls_feats
            cls_feats = self.pooler(x)

            ret = {
                "text_feats": text_feats,
                "image_feats": image_feats,
                "cls_feats": cls_feats,
                "raw_cls_feats": x[:, 0],
                "image_masks": image_masks,
                "text_masks": text_masks,
            }
        
        return ret

    def forward(self, batch):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret
        
        if "ehr_ecg_cxr" in self.current_tasks:
            ret.update(objectives.compute_ehr_ecg_cxr(self, batch))
        
        return ret

    # ===================== Training, validation and testing steps (loss and metrics computation) ===================== #
    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k]) # loss to display on progress bars
        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)                                   # compute and log epoch-level metrics

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)                                   # compute and log epoch-level metrics
    
    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()                                                    # empty dict as it is a test step
        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]
        vilt_utils.epoch_wrapup(self)                                   # compute and log metrics computed over the entire test set

    # ===================== Optimizer and lr scheduler ===================== #
    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)                            # AdamW optimizer and lr scheduler