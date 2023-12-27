"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.ostrack.backbone_txt import build_backbone_txt
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh


class OSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer_img, transformer_txt, box_head, score_head=None,
                 aux_loss=False, head_type="CORNER", run_score_head=False):
        super().__init__()
        self.backbone_img = transformer_img
        self.backbone_txt = transformer_txt
        self.box_head = box_head
        self.aux_loss = aux_loss
        self.head_type = head_type

        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                text_ids=None,
                text_masks=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        if self.backbone_txt is not None:
            src_txt, _ = self.backbone_txt(text_ids.squeeze(), text_masks.squeeze())  # [30,40,256]

        else:
            src_txt = None
        x, aux_dict = self.backbone_img(z=template,
                                        x=search,
                                        text_src=src_txt,
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def init_nlp(self, text_ids=None, text_masks=None, ):
        if self.backbone_txt is not None:
            self.src_txt, _ = self.backbone_txt(text_ids, text_masks)  # [30,40,256]
        else:
            self.src_txt = None

    def track(self, template: torch.Tensor, search: torch.Tensor, ce_template_mask=None, ce_keep_rate=None, return_last_attn=False):

        x, aux_dict = self.backbone_img(z=template,
                                        x=search,
                                        text_src=self.src_txt,
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, )

        # Forward head
        out = self.forward_head(x, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None, run_score_head=False):
        """
        cat_feature: output embeddings of the backbone_img, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, :self.feat_len_s]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new, 'score_map': score_map}

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new, 'score_map': score_map_ctr,
                   'size_map': size_map, 'offset_map': offset_map}
        else:
            raise NotImplementedError

        return out


def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone_img = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone_img.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone_img = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                               ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                               ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                               )
        hidden_dim = backbone_img.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone_img = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                                ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                                ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                                )

        hidden_dim = backbone_img.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone_img.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)
    if cfg.MODEL.LANGUAGE.ENABLE:
        backbone_txt = build_backbone_txt(cfg, hidden_dim)
    else:
        backbone_txt = None

    model = OSTrack(
        backbone_img,
        backbone_txt,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
