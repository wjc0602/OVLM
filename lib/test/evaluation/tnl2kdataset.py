import glob
import re

import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
import os
from lib.test.utils.load_text import load_text


class TNL2kDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.tnl2k_path
        self.sequence_list = [f.split('/')[-1] for f in glob.glob(os.path.join(self.base_path, '*/*'))]

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(seq_name) for seq_name in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = glob.glob(os.path.join(self.base_path, f'*/{sequence_name}'))[-1]
        anno_path = '{}/groundtruth.txt'.format(sequence_path)
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        nlp_path = '{}/language.txt'.format(sequence_path)
        nlp = open(nlp_path).readlines()[0]

        frames_path = '{}/imgs'.format(sequence_path)
        frame_list = [frame for frame in os.listdir(frames_path)]
        frame_list.sort(key=lambda f: re.sub("\D", "", f))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'tnl2k', ground_truth_rect.reshape(-1, 4), nlp=nlp)

    def __len__(self):
        return len(self.sequence_list)
