import torch
import torch.nn as nn
from torch.nn import functional as F
from .anchor_head import AnchorHead
from ...ops.roi_align import RoIAlign
from ...ops.psroi_pool import PSRoIPoolAfterPointwiseConv
from ...core import xcorr_depthwise, xcorr_fast
from ..registry import HEADS
from mmdet.ops import nms
from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox, force_fp32, bbox2delta,
                        multi_apply, multiclass_nms, bbox_overlaps)
from ..losses import accuracy
from ..builder import build_loss

class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=1, head_kernel_size=3, padding=0):
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
            nn.Conv2d(hidden, hidden, kernel_size=head_kernel_size, padding=int((head_kernel_size-1)/2), bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

        self.padding = padding

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel, padding=self.padding)
        out = self.head(feature)
        return out

class DepthwiseRPN(RPN):
    def __init__(self, in_channels, out_channels, kernel_size, target_size, n_classes, use_attention=False):
        super(DepthwiseRPN, self).__init__()
        self.rpn_feat = DepthwiseXCorr(in_channels, out_channels, out_channels, padding=0)
        sz_after_conv = target_size-kernel_size+1
        self.n_classes = n_classes
        self.rpn_cls = nn.Linear(out_channels*sz_after_conv*sz_after_conv, n_classes)
        self.rpn_reg = nn.Linear(out_channels*sz_after_conv*sz_after_conv, 4)

        self.use_attention = use_attention
        if self.use_attention:
            self.attention = nn.Linear(out_channels * sz_after_conv * sz_after_conv, sz_after_conv * sz_after_conv)

    def forward(self, z_f, x_f):
        wz = z_f.size()[-2] * z_f.size()[-1]
        wx = x_f.size()[-2] * x_f.size()[-1]
        z_f = z_f/wx
        x_f = x_f/wz
        rpn_feat = self.rpn_feat(z_f, x_f)
        N,C = rpn_feat.size()[:2]
        if self.use_attention:
            # Attention.
            rpn_feat = rpn_feat.view(N, C, -1)
            attention = self.attention(rpn_feat.view(N, -1).detach()).view(N, 1, -1)
            rpn_feat = rpn_feat*attention
        rpn_feat = rpn_feat.view(N, -1)
        cls = self.rpn_cls(rpn_feat)
        loc = self.rpn_reg(rpn_feat)
        return cls, loc

@HEADS.register_module
class SiameseRPNHead(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[7], target_sizes=[19], feat_strides=[8],
                 target_means=[.0, .0, .0, .0],
                 target_stds=[1.0, 1.0, 1.0, 1.0],
                 reg_class_agnostic = True,
                 loss_cls=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 psroi_align_kernel=False):
        super(SiameseRPNHead, self).__init__()
        self.reg_class_agnostic = reg_class_agnostic
        self.psroi_align_kernel = psroi_align_kernel
        # spatial_scales is 1./feat_strides
        self.feat_strides = feat_strides
        assert len(self.feat_strides)==1, 'There should be only 1 level for Siamese RPN.'
        spatial_scales = [1. / feat_stride for feat_stride in feat_strides]
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)

        self.target_means = target_means
        self.target_stds = target_stds

        self.kernel_crop_modules = [self._get_kernel_crop_modules(in_channels, kernel_size, spatial_scale) for kernel_size, spatial_scale in zip(kernel_sizes, spatial_scales)]
        self.target_crop_modules = [self._get_target_crop_modules(in_channels, target_size, spatial_scale) for target_size, spatial_scale in zip(target_sizes, spatial_scales)]
        self.rpn_module = DepthwiseRPN(in_channels, out_channels, kernel_sizes[0], target_sizes[0], n_classes=1)

        self.kernel_sizes = kernel_sizes
        self.target_sizes = target_sizes
        self.n_lvls = len(feat_strides)
        assert len(kernel_sizes)==1 and len(target_sizes)==1, (len(kernel_sizes), len(target_sizes))

    def _get_kernel_crop_modules(self, in_channels, kernel_size, spatial_scale):
        if self.psroi_align_kernel:
            kernel_crop_channels = 10
            kernel_crop_module = \
                PSRoIPoolAfterPointwiseConv(in_channels, kernel_crop_channels*kernel_size*kernel_size, kernel_size, spatial_scale,n_prev=2)
            kernel_crop_module.add_post_module(
                                  nn.Sequential(nn.ReLU(inplace=True),
                                                nn.Conv2d(kernel_crop_channels,
                                                          in_channels,
                                                          kernel_size=1, stride=1, bias=False),
                                                nn.BatchNorm2d(in_channels),)
                                               )
        else:
            kernel_crop_module = RoIAlign(kernel_size, spatial_scale)
        return kernel_crop_module.cuda()

    def _get_target_crop_modules(self, in_channels, kernel_size, spatial_scale):
        target_crop_module = RoIAlign(kernel_size, spatial_scale)
        return target_crop_module.cuda()

    def _get_resized_kernels_targets_single_lvl(self, feat1, feat2, img_meta, feat_stride, kernel_size, target_size, rpn_rois,
                                        kernel_crop_module, target_crop_module):
        assert feat2.shape[-1]>target_size and feat2.shape[-2]>target_size, \
            'feature size:{} should be larger than target size:{}.'.format(feat2.shape[-2:], target_size)
        _, _, h, w = feat1.shape
        rpn_rois_center_x = (rpn_rois[:, 1] + rpn_rois[:, 3]) / 2.0
        rpn_rois_center_y = (rpn_rois[:, 2] + rpn_rois[:, 4]) / 2.0

        roi_kernels = torch.zeros_like(rpn_rois)
        roi_targets = torch.zeros_like(rpn_rois)
        roi_kernels[:, 0] = rpn_rois[:, 0]
        roi_targets[:, 0] = rpn_rois[:, 0]

        # kernels
        roi_kernels[:, 1] = rpn_rois[:, 1]
        roi_kernels[:, 3] = rpn_rois[:, 3]
        roi_kernels[:, 2] = rpn_rois[:, 2]
        roi_kernels[:, 4] = rpn_rois[:, 4]

        ratios = [None, None]
        ratios[0] = (roi_kernels[:, 3] - roi_kernels[:, 1]) / (kernel_size-1)
        ratios[1] = (roi_kernels[:, 4] - roi_kernels[:, 2]) / (kernel_size-1)

        # targets
        rx = (target_size-1)*ratios[0]/2.0
        ry = (target_size-1)*ratios[1]/2.0
        roi_targets[:, 1] = rpn_rois_center_x - rx
        roi_targets[:, 3] = rpn_rois_center_x + rx
        roi_targets[:, 2] = rpn_rois_center_y - ry
        roi_targets[:, 4] = rpn_rois_center_y + ry

        kernels = kernel_crop_module(feat1, roi_kernels)
        targets = target_crop_module(feat2, roi_targets)
        target_ranges = roi_targets
        target_metas = [img_meta[int(roi[0].item())] for roi in rpn_rois]

        return kernels, targets, target_ranges, target_metas


    def forward_single(self, feat1, feat2, img_meta, feat_stride, kernel_size, target_size, rpn_rois_1, kernel_crop_module,
                                                                               target_crop_module):
        kernels, targets, target_ranges, target_metas = self._get_resized_kernels_targets_single_lvl(feat1, feat2, img_meta, feat_stride,
                                                                               kernel_size, target_size,
                                                                               rpn_rois_1, kernel_crop_module,
                                                                               target_crop_module)
        cls_score, bbox_pred = self.rpn_module(kernels, targets)
        # convert to [0-1].
        cls_score = F.sigmoid(cls_score)
        return cls_score, bbox_pred, target_ranges, target_metas

    def forward(self, feat1s, feat2s, rpn_rois_1, img_metas):
        return multi_apply(self.forward_single, feat1s, feat2s, [img_metas]*self.n_lvls, self.feat_strides, self.kernel_sizes, self.target_sizes,
                           [rpn_rois_1]*self.n_lvls, self.kernel_crop_modules, self.target_crop_modules)

    def get_target(self, rpn_rois, gt_bboxes, gt_labels, cfg):
        labels = rpn_rois.new_zeros(len(gt_labels), dtype=torch.long)
        label_weights = rpn_rois.new_zeros(len(gt_labels))
        bbox_targets = rpn_rois.new_zeros(len(gt_labels), 4)
        bbox_weights = rpn_rois.new_zeros(len(gt_labels), 4)
        for idx in range(len(gt_labels)):
            if gt_labels[idx] is not None:
                labels[idx] = 1.0
                label_weights[idx] = 1.0
                pos_bbox_targets = bbox2delta(rpn_rois[idx:idx+1, 1:], gt_bboxes[idx], self.target_means,
                                              self.target_stds)
                bbox_targets[idx,:] = pos_bbox_targets
                bbox_weights[idx,:] = 1.0
            else:
                labels[idx] = 0.0
                label_weights[idx] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def one_hot(self, y, n_cls):
        batch_size = y.size()[-1]
        # One hot encoding buffer that you create out of the loop and just keep reusing
        y_onehot = y.new_zeros(batch_size, n_cls)
        y_onehot.scatter_(-1, y.view(-1,1), value=1)
        return y_onehot

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             rpn_rois,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        pos_inds = labels > 0
        pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
        if len(pos_bbox_pred) > 0:
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)

        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)

        all_boxes = delta2bbox(rpn_rois[:, 1:], bbox_pred, self.target_means, self.target_stds, None)
        bboxes = all_boxes[pos_inds]
        labels = labels.float()
        if len(bboxes)>0:
            gtbboxes = delta2bbox(rpn_rois[:, 1:], bbox_targets, self.target_means, self.target_stds, None)[pos_inds]
            iou_target = bbox_overlaps(bboxes, gtbboxes, 'iou', is_aligned=True)
            labels[pos_inds] = iou_target

        losses['loss_cls'] = self.loss_cls(
            cls_score,
            labels.view(-1, 1),
            label_weights.view(-1, 1),
            avg_factor=avg_factor,
            reduction_override=reduction_override)

        pred_bboxes = torch.cat([all_boxes, cls_score], dim=-1)

        if 'loss_bbox' in losses.keys():
            return dict(loss_siamese_rpn_cls=losses['loss_cls'],
                        loss_siamese_rpn_bbox=losses['loss_bbox']), \
                   pred_bboxes
        else:
            return dict(loss_siamese_rpn_cls=losses['loss_cls'],
                        loss_siamese_rpn_bbox=losses['loss_cls'].new_zeros(
                            losses['loss_cls'].shape)), \
                   pred_bboxes


    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   n_batches,
                   rois,
                   cls_scores,
                   bbox_preds,
                   target_metas,
                   cfg=None,):
        assert len(cls_scores)==1 and len(bbox_preds)==1 and len(target_metas)==1
        cls_scores = cls_scores[0]
        bbox_preds = bbox_preds[0]
        target_metas = target_metas[0]
        bboxes_list = [[] for _ in range(n_batches)]
        scores_list = [[] for _ in range(n_batches)]
        for roi, cls_score, bbox_pred, target_meta in zip(rois, cls_scores, bbox_preds, target_metas):
            roi = roi.view(1, -1)
            cls_score = cls_score.view(1, -1)
            bbox_pred = bbox_pred.view(1, -1)
            img_shape = target_meta['img_shape']

            if isinstance(cls_score, list):
                cls_score = sum(cls_score) / float(len(cls_score))
            #scores = F.softmax(cls_score, dim=1) if cls_score is not None else None
            scores = cls_score
            if bbox_pred is not None:
                bboxes = delta2bbox(roi[:, 1:], bbox_pred, self.target_means,
                                    self.target_stds, img_shape)
            else:
                bboxes = roi[:, 1:].clone()
                if img_shape is not None:
                    bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                    bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)
            bboxes_list[int(roi[0, 0])].append(bboxes)
            scores_list[int(roi[0, 0])].append(scores)
        bboxes = [torch.cat(bboxes, dim=0) for bboxes in bboxes_list]
        scores = [torch.cat(scores, dim=0) for scores in scores_list]
        return bboxes, scores

    def get_rois_from_boxes(self, n_batches, bboxes_list, scores_list, score_threshold=0.):
        rpn_roi_list = []
        for i_batch in range(n_batches):
            bboxes = bboxes_list[i_batch]
            scores = scores_list[i_batch].view(-1)
            rpn_rois = bboxes.new_zeros((bboxes.shape[0], bboxes.shape[1]+1))
            rpn_rois[:, :4] = bboxes
            rpn_rois[:, 4] = scores
            if score_threshold>0:
                inds = scores>score_threshold
                rpn_rois = rpn_rois[inds,:]
            rpn_roi_list.append(rpn_rois)
        return rpn_roi_list




