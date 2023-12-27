"""
Basic OSTrack model.
"""
import os

import torch
from einops import rearrange
from timm.models.layers import trunc_normal_

from torch import nn
from torch.nn.modules.transformer import _get_clones
import torchvision
import torch.nn.functional as F

from lib.models.layers.head import build_box_head, MLP
from lib.models.ostrack.backbone_txt import build_backbone_txt
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh


class ScoreDecoder(nn.Module):
    """ This is the base class for Transformer Tracking """

    def __init__(self, hidden_dim, pool_size=4, mode='multimodal'):
        """ Initializes the model.
        """
        super().__init__()
        self.num_heads = 12
        self.pool_size = pool_size
        self.head = MLP(hidden_dim, hidden_dim, 1, 3)
        self.scale = hidden_dim ** -0.5
        self.mode = mode

        self.proj_q = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))
        self.proj_k = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))
        self.proj_v = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))

        self.proj = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(2))

        self.score_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        trunc_normal_(self.score_token, std=.02)

    def forward(self, cat_feature, search_box, feat_sz_s):
        """
        :param search_box: with normalized coords. (x0, y0, x1, y1)
        :return:
        """

        search_feat = cat_feature[:, :self.feat_len_s].clone()

        search_feat = rearrange(search_feat, 'b (w h) d -> b d w h', w=feat_sz_s, h=feat_sz_s)
        b, c, h, w = search_feat.shape
        search_box = search_box.clone() * feat_sz_s
        bb_pool = search_box.view(-1, 4)
        batch_size = bb_pool.shape[0]

        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(bb_pool.device)
        target_roi = torch.cat((batch_index, bb_pool), dim=1)
        search_feat = rearrange(torchvision.ops.roi_align(input=search_feat, boxes=target_roi, output_size=4), ' b d w h -> b (w h) d', )

        # template_feat = torch.cat([template_feat, language_feat], dim=1)
        if self.mode == 'multimodal':
            template_feat = cat_feature[:, self.feat_len_s:].clone()
        elif self.mode == 'language':
            template_feat = cat_feature[:, self.feat_len_s:self.feat_len_s + 40].clone()
        elif self.mode == 'memory':
            template_feat = cat_feature[:, self.feat_len_s + 40:].clone()

        # decoder1: query for search_box feat
        # decoder2: query for template feat
        x = self.score_token.expand(b, -1, -1)
        x = self.norm1(x)
        # search_box_feat = rearrange(torchvision.ops.roi_align(search_feat, target_roi, output_size=4), 'b c h w -> b (h w) c')
        kv_memory = [search_feat, template_feat]
        for i in range(2):
            q = rearrange(self.proj_q[i](x), 'b t (n d) -> b n t d', n=self.num_heads)
            k = rearrange(self.proj_k[i](kv_memory[i]), 'b t (n d) -> b n t d', n=self.num_heads)
            v = rearrange(self.proj_v[i](kv_memory[i]), 'b t (n d) -> b n t d', n=self.num_heads)

            attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
            attn = F.softmax(attn_score, dim=-1)
            x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
            x = rearrange(x, 'b h t d -> b t (h d)')  # (b, 1, c)
            x = self.proj[i](x)
            x = self.norm2[i](x)
        out_scores = self.head(x)  # (b, 1, 1)

        return out_scores


class OSTrackRE(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer_img, transformer_txt, box_head, score_head,
                 aux_loss=False, head_type="CORNER", run_score_head=False):
        super().__init__()
        self.backbone_img = transformer_img
        self.backbone_txt = transformer_txt
        self.box_head = box_head
        self.aux_loss = aux_loss
        self.head_type = head_type

        self.score_head = score_head
        self.run_score_head = run_score_head

        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

            self.score_head.feat_len_s = self.feat_len_s

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                text_ids=None,
                text_masks=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False
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
        out = self.forward_head(feat_last, None, self.run_score_head)

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

        out = self.forward_head(x, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):

        enc_opt = cat_feature[:, :self.feat_len_s]
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
            out = {'pred_boxes': outputs_coord_new, 'score_map': score_map_ctr, 'size_map': size_map, 'offset_map': offset_map}
        else:
            raise NotImplementedError

        # run score head
        if self.run_score_head:
            # if True:
            pred_score = self.score_head(cat_feature.clone(), outputs_coord_new, self.feat_sz_s).squeeze()
            out.update({'pred_score': pred_score})

        return out


def build_ostrack_re(cfg, training=True):
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

    score_head = ScoreDecoder(hidden_dim=hidden_dim, mode=cfg.MODEL.RESULT_EVAL.TYPE)  # the proposed score prediction module (SPM)

    model = OSTrackRE(
        backbone_img,
        backbone_txt,
        box_head,
        score_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        run_score_head=cfg.MODEL.RESULT_EVAL.ENABLE
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

        print("missing keys:", missing_keys)
        print("unexpected keys:", unexpected_keys)
        print("Loading pretrained mixformer weights done.")

    return model
