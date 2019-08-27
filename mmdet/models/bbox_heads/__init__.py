from .bbox_head import BBoxHead
from .convfc_bbox_head import ConvFCBBoxHead, SharedFCBBoxHead
from .double_bbox_head import DoubleConvFCBBoxHead
from .avepool2d_bbox_head import Avepool2dBoxHead

__all__ = [
    'BBoxHead', 'ConvFCBBoxHead', 'SharedFCBBoxHead', 'DoubleConvFCBBoxHead','Avepool2dBoxHead'
]
