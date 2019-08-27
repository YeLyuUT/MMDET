import torch.nn as nn

from ..registry import HEADS
from .bbox_head import BBoxHead

@HEADS.register_module
class Avepool2dBoxHead(BBoxHead):
    def __init__(self, *args,**kargs):
        super(Avepool2dBoxHead, self).__init__(*args,**kargs)
        self.avepool = nn.AvgPool2d(kernel_size=self.roi_feat_size, stride=self.roi_feat_size)

    def forward(self, feature_tuple):
        feature_cls, feature_box = feature_tuple
        cls_score = self.avepool(feature_cls).squeeze(-1).squeeze(-1)
        box_pred = self.avepool(feature_box).squeeze(-1).squeeze(-1)
        return cls_score, box_pred