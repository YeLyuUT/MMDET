import numpy as np

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class ImageNetDETVIDDataset(CustomDataset):

    CLASSES = ('airplane','antelope','bear','bicycle','bird','bus',
               'car','cattle','dog','domestic_cat','elephant','fox',
               'giant_panda','hamster','horse','lion','lizard','monkey',
               'motorcycle','rabbit','red_panda','sheep','snake','squirrel',
               'tiger','train','turtle','watercraft','whale','zebra')

    def get_ann_info(self, idx):
        ann = self.img_infos[idx]['ann']
        # modify type if necessary.
        if not isinstance(ann['bboxes'],np.ndarray):
            ann['bboxes'] = np.array(ann['bboxes'], dtype=np.float32).reshape(-1, 4)
        if not isinstance(ann['labels'], np.ndarray):
            ann['labels'] = np.array(ann['labels'], dtype=np.int64)#.reshape(-1, 1)
        self.img_infos[idx]['ann']=ann
        return ann