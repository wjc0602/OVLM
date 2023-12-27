import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class OTBDataset(BaseDataset):

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.otb_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                                                                           sequence_path=sequence_path,
                                                                           frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame + init_omit, end_frame + 1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        # NOTE: OTB has some weird annos which panda cannot handle
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        nlp_path = '{}/{}'.format(self.base_path, sequence_info['nlp_path'])
        nlp = open(nlp_path).readlines()[0].replace('\n', '')

        return Sequence(sequence_info['name'], frames, 'otb', ground_truth_rect[init_omit:, :], nlp=nlp, object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "Biker", "path": "OTB_videos/Biker/img", "startFrame": 1, "endFrame": 142, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Biker/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Biker.txt", "object_class": "person head"},
            {"name": "Bird1", "path": "OTB_videos/Bird1/img", "startFrame": 1, "endFrame": 408, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Bird1/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Bird1.txt", "object_class": "bird"},
            {"name": "Bird2", "path": "OTB_videos/Bird2/img", "startFrame": 1, "endFrame": 99, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Bird2/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Bird2.txt", "object_class": "bird"},

            {"name": "BlurBody", "path": "OTB_videos/BlurBody/img", "startFrame": 1, "endFrame": 334, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/BlurBody/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/BlurBody.txt", "object_class": "person"},
            {"name": "BlurCar1", "path": "OTB_videos/BlurCar1/img", "startFrame": 247, "endFrame": 988, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/BlurCar1/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/BlurCar1.txt", "object_class": "car"},
            {"name": "BlurCar2", "path": "OTB_videos/BlurCar2/img", "startFrame": 1, "endFrame": 585, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/BlurCar2/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/BlurCar2.txt", "object_class": "car"},
            {"name": "BlurCar3", "path": "OTB_videos/BlurCar3/img", "startFrame": 3, "endFrame": 359, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/BlurCar3/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/BlurCar3.txt", "object_class": "car"},
            {"name": "BlurCar4", "path": "OTB_videos/BlurCar4/img", "startFrame": 18, "endFrame": 397, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/BlurCar4/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/BlurCar4.txt", "object_class": "car"},

            {"name": "BlurFace", "path": "OTB_videos/BlurFace/img", "startFrame": 1, "endFrame": 493, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/BlurFace/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/BlurFace.txt", "object_class": "face"},
            {"name": "BlurOwl", "path": "OTB_videos/BlurOwl/img", "startFrame": 1, "endFrame": 631, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/BlurOwl/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/BlurOwl.txt", "object_class": "other"},
            {"name": "Board", "path": "OTB_videos/Board/img", "startFrame": 1, "endFrame": 698, "nz": 5, "ext": "jpg", "anno_path": "OTB_videos/Board/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Board.txt", "object_class": "other"},
            {"name": "Bolt2", "path": "OTB_videos/Bolt2/img", "startFrame": 1, "endFrame": 293, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Bolt2/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Bolt2.txt", "object_class": "person"},
            {"name": "Box", "path": "OTB_videos/Box/img", "startFrame": 1, "endFrame": 1161, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Box/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Box.txt", "object_class": "other"},

            {"name": "Car1", "path": "OTB_videos/Car1/img", "startFrame": 1, "endFrame": 1020, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Car1/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Car1.txt", "object_class": "car"},
            {"name": "Car2", "path": "OTB_videos/Car2/img", "startFrame": 1, "endFrame": 913, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Car2/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Car2.txt", "object_class": "car"},
            {"name": "Car24", "path": "OTB_videos/Car24/img", "startFrame": 1, "endFrame": 3059, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Car24/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Car24.txt", "object_class": "car"},
            {"name": "Coupon", "path": "OTB_videos/Coupon/img", "startFrame": 1, "endFrame": 327, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Coupon/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Coupon.txt", "object_class": "other"},
            {"name": "Crowds", "path": "OTB_videos/Crowds/img", "startFrame": 1, "endFrame": 347, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Crowds/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Crowds.txt", "object_class": "person"},

            {"name": "Dancer", "path": "OTB_videos/Dancer/img", "startFrame": 1, "endFrame": 225, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Dancer/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Dancer.txt", "object_class": "person"},
            {"name": "Dancer2", "path": "OTB_videos/Dancer2/img", "startFrame": 1, "endFrame": 150, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Dancer2/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Dancer2.txt", "object_class": "person"},
            {"name": "Diving", "path": "OTB_videos/Diving/img", "startFrame": 1, "endFrame": 215, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Diving/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Diving.txt", "object_class": "person"},
            {"name": "Dog", "path": "OTB_videos/Dog/img", "startFrame": 1, "endFrame": 127, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Dog/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Dog.txt", "object_class": "dog"},
            {"name": "DragonBaby", "path": "OTB_videos/DragonBaby/img", "startFrame": 1, "endFrame": 113, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/DragonBaby/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/DragonBaby.txt", "object_class": "face"},

            {"name": "Girl2", "path": "OTB_videos/Girl2/img", "startFrame": 1, "endFrame": 1500, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Girl2/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Girl2.txt", "object_class": "person"},
            {"name": "Gym", "path": "OTB_videos/Gym/img", "startFrame": 1, "endFrame": 767, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Gym/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Gym.txt", "object_class": "person"},
            {"name": "Human2", "path": "OTB_videos/Human2/img", "startFrame": 1, "endFrame": 1128, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Human2/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Human2.txt", "object_class": "person"},
            {"name": "Human3", "path": "OTB_videos/Human3/img", "startFrame": 1, "endFrame": 1698, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Human3/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Human3.txt", "object_class": "person"},
            {"name": "Human4", "path": "OTB_videos/Human4/img", "startFrame": 1, "endFrame": 667, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Human4/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Human4.txt", "object_class": "person"},

            {"name": "Human5", "path": "OTB_videos/Human5/img", "startFrame": 1, "endFrame": 713, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Human5/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Human5.txt", "object_class": "person"},
            {"name": "Human6", "path": "OTB_videos/Human6/img", "startFrame": 1, "endFrame": 792, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Human6/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Human6.txt", "object_class": "person"},
            {"name": "Human7", "path": "OTB_videos/Human7/img", "startFrame": 1, "endFrame": 250, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Human7/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Human7.txt", "object_class": "person"},
            {"name": "Human8", "path": "OTB_videos/Human8/img", "startFrame": 1, "endFrame": 128, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Human8/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Human8.txt", "object_class": "person"},
            {"name": "Human9", "path": "OTB_videos/Human9/img", "startFrame": 1, "endFrame": 305, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Human9/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Human9.txt", "object_class": "person"},

            {"name": "Jump", "path": "OTB_videos/Jump/img", "startFrame": 1, "endFrame": 122, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Jump/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Jump.txt", "object_class": "person"},
            {"name": "KiteSurf", "path": "OTB_videos/KiteSurf/img", "startFrame": 1, "endFrame": 84, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/KiteSurf/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/KiteSurf.txt", "object_class": "face"},
            {"name": "Man", "path": "OTB_videos/Man/img", "startFrame": 1, "endFrame": 134, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Man/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Man.txt", "object_class": "face"},
            {"name": "Panda", "path": "OTB_videos/Panda/img", "startFrame": 1, "endFrame": 1000, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Panda/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Panda.txt", "object_class": "mammal"},
            {"name": "RedTeam", "path": "OTB_videos/RedTeam/img", "startFrame": 1, "endFrame": 1918, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/RedTeam/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/RedTeam.txt", "object_class": "vehicle"},

            {"name": "Rubik", "path": "OTB_videos/Rubik/img", "startFrame": 1, "endFrame": 1997, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Rubik/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Rubik.txt", "object_class": "other"},
            {"name": "Skater", "path": "OTB_videos/Skater/img", "startFrame": 1, "endFrame": 160, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Skater/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Skater.txt", "object_class": "person"},
            {"name": "Skater2", "path": "OTB_videos/Skater2/img", "startFrame": 1, "endFrame": 435, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Skater2/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Skater2.txt", "object_class": "person"},
            {"name": "Skating2-1", "path": "OTB_videos/Skating2-1/img", "startFrame": 1, "endFrame": 473, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Skating2-1/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Skating2-1.txt", "object_class": "person"},
            {"name": "Skating2-2", "path": "OTB_videos/Skating2-2/img", "startFrame": 1, "endFrame": 473, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Skating2-2/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Skating2-2.txt", "object_class": "person"},

            {"name": "Surfer", "path": "OTB_videos/Surfer/img", "startFrame": 1, "endFrame": 376, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Surfer/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Surfer.txt", "object_class": "person head"},
            {"name": "Toy", "path": "OTB_videos/Toy/img", "startFrame": 1, "endFrame": 271, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Toy/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Toy.txt", "object_class": "other"},
            {"name": "Trans", "path": "OTB_videos/Trans/img", "startFrame": 1, "endFrame": 124, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Trans/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Trans.txt", "object_class": "other"},
            {"name": "Twinnings", "path": "OTB_videos/Twinnings/img", "startFrame": 1, "endFrame": 472, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Twinnings/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Twinnings.txt", "object_class": "other"},
            {"name": "Vase", "path": "OTB_videos/Vase/img", "startFrame": 1, "endFrame": 271, "nz": 4, "ext": "jpg", "anno_path": "OTB_videos/Vase/groundtruth_rect.txt",
             "nlp_path": "OTB_query_test/Vase.txt", "object_class": "other"},

        ]

        return sequence_info_list
