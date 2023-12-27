import math

import numpy as np
from transformers import RobertaTokenizer

from lib.models.ostrack import build_ostrack_re
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import matplotlib.pyplot as plt
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class OSTrackRE(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrackRE, self).__init__(params)
        network = build_ostrack_re(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()

        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.run_score = self.cfg.MODEL.RESULT_EVAL.ENABLE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        self.all_memory_frames = []
        self.all_memory_imgs = []

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        self.memory_num = 0

        self.hp_template_num = params.cfg.TEST.ONLINE_SIZES[dataset_name.upper()]
        self.hp_gpu_memory_threshold = 3000
        self.hp_confidence_threshold = 0.9

        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes

    def initialize(self, image, nlp, info: dict, seq_name):
        if self.debug:
            self.save_dir = f"debug_lasot_mts/{seq_name}_{nlp}"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # add the first template into memory
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor, output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        self.all_memory_frames.append(template)
        self.all_memory_imgs.append(z_patch_arr)
        self.memory_num += 1

        self.device = template.device

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor, self.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, self.device, template_bbox)

        # init nlp
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        nlp_encoded = self.tokenizer.batch_encode_plus([nlp], max_length=40, padding="max_length", return_tensors='pt')
        text_ids = nlp_encoded["input_ids"].to(self.device)
        text_masks = nlp_encoded["attention_mask"].to(self.device)
        self.network.init_nlp(text_ids, text_masks)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):

        # select memory
        if self.frame_id <= self.hp_template_num:
            templates = self.all_memory_frames
        else:
            templates, select_indexs = self.select_representatives(self.all_memory_frames)

        # track
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor, output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.track(template=templates.copy(), search=x_dict, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']

        if self.cfg.MODEL.RESULT_EVAL.ENABLE:
            re_score = out_dict['pred_score'].sigmoid().item()
        else:
            re_score = 1

        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # add new frame into memory
        if re_score > self.hp_confidence_threshold:
            bbox_int = [int(x) for x in self.state]
            z_patch_arr, resize_factor, z_amask_arr = sample_target(image, bbox_int, self.params.template_factor, output_sz=self.params.template_size)
            template = self.preprocessor.process(z_patch_arr, z_amask_arr)

            if self.frame_id > self.hp_gpu_memory_threshold:
                template = template.detach().cpu()
            self.all_memory_frames.append(template)
            self.all_memory_imgs.append(z_patch_arr)
            self.memory_num += 1

        # for debug
        if self.debug:
            # draw bbox
            # x1, y1, w, h = self.state
            # image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            # save_path = os.path.join(self.save_dir, "%04d_%04f.jpg" % (self.frame_id, re_score))
            # cv2.imwrite(save_path, image_BGR)

            # visualizate memory token select
            if self.memory_num >= 3 and self.frame_id % 1000 == 0:
                removed_indexes = out_dict['removed_indexes_s']
                select_indexs = select_indexs - 1
                for i in range(len(select_indexs)):  # for every target sample
                    img_mst = cv2.cvtColor(self.all_memory_imgs[select_indexs[i]], cv2.COLOR_RGB2BGR)
                    save_path = os.path.join(self.save_dir, f"{self.frame_id}_{i}_{0}.jpg")
                    cv2.imwrite(save_path, img_mst)
                    for j in range(len(removed_indexes)):  # for every select stage
                        indexs = removed_indexes[j]
                        for k in indexs[0]:
                            if k > i * 64 and k < (i + 1) * 64:
                                k = k % 64
                                x = int(k.item() // 8)
                                y = int(k.item() % 8)
                                img_mst[x * 16:(x + 1) * 16, y * 16:(y + 1) * 16, :] = 1
                        save_path = os.path.join(self.save_dir, f"{self.frame_id}_{i}_{j + 1}.jpg")
                        cv2.imwrite(save_path, img_mst)

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state, "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone_img.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights

    def select_representatives(self, frames):
        memory_len = len(frames)
        dur = memory_len // self.hp_template_num
        indexes = np.concatenate([np.array([1]), np.array(list(range(self.hp_template_num))) * dur + dur // 2 + 1])
        indexes = np.unique(indexes)

        representatives = []
        for idx in indexes:
            fm = frames[idx - 1]
            if not fm.is_cuda:
                fm = fm.to(self.device)
            representatives.append(fm)

        return representatives, indexes


def get_tracker_class():
    return OSTrackRE
