import numpy as np

from .custom import CustomDataset
from .custom_pair import CustomPairDataset
from .custom_block import CustomBlockDataset
from .registry import DATASETS


@DATASETS.register_module
class ImageNetDETVIDDataset(CustomDataset):

    CLASSES = ('airplane','antelope','bear','bicycle','bird','bus',
               'car','cattle','dog','domestic_cat','elephant','fox',
               'giant_panda','hamster','horse','lion','lizard','monkey',
               'motorcycle','rabbit','red_panda','sheep','snake','squirrel',
               'tiger','train','turtle','watercraft','whale','zebra')
    
    def __init__(self,*args,**kargs):
      super().__init__(*args,**kargs)
      self.img_ids = list(range(len(self.img_infos)))
      self.cat_ids = list(range(len(self.CLASSES)))

    def get_ann_info(self, idx):
        ann = self.img_infos[idx]['ann']
        # modify type if necessary.
        if not isinstance(ann['bboxes'],np.ndarray):
            ann['bboxes'] = np.array(ann['bboxes'], dtype=np.float32).reshape(-1, 4)
        if not isinstance(ann['labels'], np.ndarray):
            ann['labels'] = np.array(ann['labels'], dtype=np.int64)#.reshape(-1, 1)
        self.img_infos[idx]['ann']=ann
        return ann


@DATASETS.register_module
class ImageNetVIDBlockDataset(CustomBlockDataset):
  CLASSES = ('airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus',
             'car', 'cattle', 'dog', 'domestic_cat', 'elephant', 'fox',
             'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey',
             'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake', 'squirrel',
             'tiger', 'train', 'turtle', 'watercraft', 'whale', 'zebra')

  def __init__(self, *args, **kargs):
    super().__init__(*args, **kargs)
    self.img_ids = list(range(len(self.img_infos)))
    self.cat_ids = list(range(len(self.CLASSES)))

  def get_ann_info(self, idx):
    ann = self.img_infos[idx]['ann']
    # modify type if necessary.
    if not isinstance(ann['bboxes'], np.ndarray):
      ann['bboxes'] = np.array(ann['bboxes'], dtype=np.float32).reshape(-1, 4)
    if not isinstance(ann['labels'], np.ndarray):
      ann['labels'] = np.array(ann['labels'], dtype=np.int64)  # .reshape(-1, 1)
    self.img_infos[idx]['ann'] = ann
    return ann


@DATASETS.register_module
class ImageNetVIDPairDataset(CustomPairDataset):
  CLASSES = ('airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus',
             'car', 'cattle', 'dog', 'domestic_cat', 'elephant', 'fox',
             'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey',
             'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake', 'squirrel',
             'tiger', 'train', 'turtle', 'watercraft', 'whale', 'zebra')

  def __init__(self, *args, **kargs):
    super().__init__(*args, **kargs)
    self.img_ids = list(range(len(self.img_infos)))
    self.cat_ids = list(range(len(self.CLASSES)))

  def get_ann_info(self, idx):
    ann1 = self.img_infos[idx]['ann1']
    ann2 = self.img_infos[idx]['ann2']
    # modify type if necessary.
    if not isinstance(ann1['bboxes'], np.ndarray):
      ann1['bboxes'] = np.array(ann1['bboxes'], dtype=np.float32).reshape(-1, 4)
    if not isinstance(ann1['labels'], np.ndarray):
      ann1['labels'] = np.array(ann1['labels'], dtype=np.int64)
    if not isinstance(ann1['trackids'], np.ndarray):
      ann1['trackids'] = np.array(ann1['trackids'], dtype=np.int64)
    self.img_infos[idx]['ann1'] = ann1

    if not isinstance(ann2['bboxes'], np.ndarray):
      ann2['bboxes'] = np.array(ann2['bboxes'], dtype=np.float32).reshape(-1, 4)
    if not isinstance(ann2['labels'], np.ndarray):
      ann2['labels'] = np.array(ann2['labels'], dtype=np.int64)
    if not isinstance(ann2['trackids'], np.ndarray):
      ann2['trackids'] = np.array(ann2['trackids'], dtype=np.int64)
    self.img_infos[idx]['ann2'] = ann2
    return ann1, ann2