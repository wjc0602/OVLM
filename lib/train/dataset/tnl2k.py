import glob
import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class TNL2k(BaseVideoDataset):

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None):

        root = env_settings().lasot_dir if root is None else root
        super().__init__('TNL2k', root, image_loader)

        self.anno_files = sorted(glob.glob(os.path.join(root, '*/*/groundtruth.txt')))
        self.nlp_files = sorted(glob.glob(os.path.join(root, '*/*/language.txt')))
        self.seq_dirs = [os.path.dirname(f) for f in self.anno_files]
        self.seq_names = [os.path.basename(os.path.dirname(f)) for f in self.anno_files]

    def get_name(self):
        return 'tnl2k'

    def get_num_sequences(self):
        return len(self.seq_dirs)

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def _read_nlp(self, seq_path):
        nlp = os.path.join(seq_path, "language.txt")
        txt = open(nlp).readlines()
        return txt

    def get_sequence_info(self, seq_id):
        seq_path = self.seq_dirs[seq_id]
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        # return os.path.join(seq_path, 'imgs', '{:08}.jpg'.format(frame_id + 1))  # frames start from 1
        return sorted(glob.glob(os.path.join(seq_path, 'imgs/*')))[frame_id]  # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self.seq_dirs[seq_id]
        nlp = self._read_nlp(seq_path)
        # frame_list
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)
        # anno_frames
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta, nlp
