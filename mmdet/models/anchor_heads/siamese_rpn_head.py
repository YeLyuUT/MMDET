import torch
import torch.nn as nn
from torch.nn import functional as F
from .rpn_head import RPNHead
from ...ops.roi_align import RoIAlign
from ...core import xcorr_depthwise
from ..registry import HEADS

class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out

class DepthwiseRPN(RPNHead):
    def __init__(self, in_channels, **kwargs):
        super(DepthwiseRPN, self).__init__(in_channels, **kwargs)

    def _init_layers(self):
        self.rpn_cls = DepthwiseXCorr(self.in_channels, self.in_channels, 2 * self.num_anchors)
        self.rpn_reg = DepthwiseXCorr(self.in_channels, self.in_channels, 4 * self.num_anchors)

    def forward_single(self, z_f, x_f):
        cls = self.rpn_cls(z_f, x_f)
        loc = self.rpn_reg(z_f, x_f)
        return cls, loc

@HEADS.register_module
class SiameseRPN(nn.Module):
    def __init__(self, feat_strides, kernel_size=3, target_size=25):
        super(SiameseRPN, self).__init__()
        # spatial_scales is 1./feat_strides
        spatial_scales = [1. / s for s in feat_strides]
        self.kernel_crop_modules = self._get_kernel_crop_modules(kernel_size, spatial_scales)
        self.feat_strides = feat_strides
        self.kernel_size = kernel_size
        self.target_size = target_size

    def _get_kernel_crop_modules(self, kernel_size, spatial_scales):
        kernel_crop_modules = []
        for spatial_scale in spatial_scales:
            kernel_crop_modules.append(RoIAlign(kernel_size, spatial_scale))
        return kernel_crop_modules

    def _adjust_rpn_center(self, roi_center_x, roi_center_y, radius, min_w, max_w, min_h, max_h):
        roi_center_x = roi_center_x - torch.min(roi_center_x - radius - min_w, 0.)
        roi_center_x = roi_center_x - torch.max(roi_center_x + radius - max_w, 0.)
        roi_center_y = roi_center_y - torch.min(roi_center_y - radius - min_h, 0.)
        roi_center_y = roi_center_y - torch.max(roi_center_y + radius - max_h, 0.)
        return roi_center_x, roi_center_y

    def _crop_feature_by_rois_with_padding(self, features, feat_stride, rois):
        x1 = rois[:, 0]
        y1 = rois[:, 1]
        x2 = rois[:, 2]
        y2 = rois[:, 3]
        x1 = torch.min(x1)
        y1 = torch.min(y1)
        x2 = torch.max(x2)
        y2 = torch.max(y2)
        pad_left, pad_right, pad_up, pad_down = features.new_zeros(4)
        h,w = features.size()[2:]
        h = h*feat_stride
        w = w*feat_stride
        extra = 0 - x1
        if extra > 0:
            pad_left = torch.ceil(extra / feat_stride).type(torch.int)
        extra = 0 - y1
        if extra > 0:
            pad_up = torch.ceil(extra / feat_stride).type(torch.int)
        extra = x2-(w-1)*feat_stride
        if extra > 0:
            pad_right = torch.ceil(extra / feat_stride).type(torch.int)
        extra = y2-(h-1)*feat_stride
        if extra > 0:
            pad_down = torch.ceil(extra / feat_stride).type(torch.int)

        features = F.pad(features, pad=(pad_left, pad_right, pad_up, pad_down), mode='constant', value=0)
        return features

    def _get_kernels_targets_single_lvl(self, feat1, feat2, feat_stride, kernel_size, target_size, rpn_rois):
        _, _, h, w = feat1.shape
        rpn_rois_center_x = (rpn_rois[:, 0] + rpn_rois[:, 2]) / 2.0
        rpn_rois_center_y = (rpn_rois[:, 1] + rpn_rois[:, 3]) / 2.0
        # crop kernels
        r = (kernel_size - 1) / 2.0 * feat_stride
        rpn_for_crop = torch.zeros_like(rpn_rois)
        rpn_rois_center_x, rpn_rois_center_y = self._adjust_rpn_center(rpn_rois_center_x, rpn_rois_center_y, r, 0,
                                                                       (w - 1) * feat_stride, 0, (h - 1) * feat_stride)
        rpn_for_crop[:, 0] = rpn_rois_center_x - r
        rpn_for_crop[:, 2] = rpn_rois_center_x + r
        rpn_for_crop[:, 1] = rpn_rois_center_y - r
        rpn_for_crop[:, 3] = rpn_rois_center_y + r
        kernels = self.kernel_crop_module(feat1, rpn_for_crop)
        # crop targets
        r = (target_size - 1) / 2.0 * feat_stride
        rpn_for_crop[:, 0] = rpn_rois_center_x - r
        rpn_for_crop[:, 2] = rpn_rois_center_x + r
        rpn_for_crop[:, 1] = rpn_rois_center_y - r
        rpn_for_crop[:, 3] = rpn_rois_center_y + r
        feat2 = self._crop_feature_by_rois_with_padding(feat2, feat_stride, rpn_for_crop)
        targets = self.kernel_crop_module(feat2, rpn_for_crop)
        target_ranges = rpn_for_crop
        return kernels, targets, target_ranges


    def forward(self, feat_lvls_1, feat_lvls_2, rpn_rois_1):
        for feat1, feat2, feat_stride in zip(feat_lvls_1, feat_lvls_2, self.feat_strides):
            kernels, targets, target_ranges = self._get_kernels_targets_single_lvl(feat1, feat2, feat_stride,
                                                                                   self.kernel_size, self.target_size,
                                                                                   rpn_rois_1)
            xcorr_depthwise(targets, kernels)

