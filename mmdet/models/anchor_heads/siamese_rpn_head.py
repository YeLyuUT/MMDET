import torch
import torch.nn as nn
from torch.nn import functional as F
from .anchor_head import AnchorHead
from ...ops.roi_align import RoIAlign
from ...ops.psroi_pool import PSRoIPoolAfterPointwiseConv
from ...core import xcorr_depthwise, xcorr_fast
from ..registry import HEADS
from mmdet.ops import nms
from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox, force_fp32,
                        multi_apply, multiclass_nms)

class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=1, head_kernel_size=7, padding=0):
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
    def __init__(self, anchor_num, in_channels, out_channels, padding=0, use_sigmoid=True):
        super(DepthwiseRPN, self).__init__()
        if use_sigmoid is True:
            self.rpn_cls = DepthwiseXCorr(in_channels, out_channels, 1 * anchor_num, padding=padding)
        else:
            self.rpn_cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num, padding=padding)
        self.rpn_reg = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num, padding=padding)

    def forward(self, z_f, x_f):
        wz = z_f.size()[-2] * z_f.size()[-1]
        wx = x_f.size()[-2] * x_f.size()[-1]
        z_f = z_f/wx
        x_f = x_f/wz
        cls = self.rpn_cls(z_f, x_f)
        loc = self.rpn_reg(z_f, x_f)
        return cls, loc

class GroupCorr(nn.Module):
    def __init__(self, inchannels, hidden, out_channels, n_group, kernel_size=3, anchor_num=5, padding=0):
        super(GroupCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(inchannels, hidden*n_group, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden*n_group),
            nn.ReLU(inplace=True),
            )
        self.conv_search = nn.Sequential(
            nn.Conv2d(inchannels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
                nn.Conv2d(n_group, n_group, kernel_size=1, bias=False),
                nn.BatchNorm2d(n_group),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_group, out_channels, kernel_size=1)
                )
        self.padding = padding

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        middle_feat = xcorr_fast(search, kernel, self.padding)
        out = self.head(middle_feat)
        return out

class GroupCorrRPN(RPN):
    def __init__(self, in_channels=256, out_channels=256, n_group=32, anchor_num=5, padding=0, use_sigmoid=True):
        super(GroupCorrRPN, self).__init__()
        if use_sigmoid is True:
            self.cls = GroupCorr(in_channels, out_channels, 1 * anchor_num, n_group,3, anchor_num, padding=padding)
        else:
            self.cls = GroupCorr(in_channels, out_channels, 2 * anchor_num, n_group,3, anchor_num, padding=padding)
        self.loc = GroupCorr(in_channels, out_channels, 4 * anchor_num, n_group,3, anchor_num, padding=padding)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc

@HEADS.register_module
class SiameseRPNHead(AnchorHead):
    def __init__(self, in_channels, out_channels, kernel_sizes=[7], target_sizes=[25], **kwargs):

        super(SiameseRPNHead, self).__init__(2, in_channels, **kwargs)
        # spatial_scales is 1./feat_strides
        assert len(self.anchor_strides)==1, 'There should be only 1 level for Siamese RPN.'
        feat_strides = self.anchor_strides
        spatial_scales = [1. / feat_stride for feat_stride in feat_strides]
        self.kernel_crop_modules = [self._get_kernel_crop_modules(in_channels, kernel_size, spatial_scale) for kernel_size, spatial_scale in zip(kernel_sizes, spatial_scales)]
        self.target_crop_modules = [self._get_target_crop_modules(in_channels, target_size, spatial_scale) for target_size, spatial_scale in zip(target_sizes, spatial_scales)]
        self.rpn_module = DepthwiseRPN(self.num_anchors, in_channels, out_channels, padding = int((kernel_sizes[0]-1)/2), use_sigmoid=self.use_sigmoid_cls)
        #self.rpn_module = GroupCorrRPN(in_channels, out_channels,32,self.num_anchors,padding=int((kernel_sizes[0] - 1) / 2), use_sigmoid=self.use_sigmoid_cls)
        self.feat_strides = feat_strides
        self.kernel_sizes = kernel_sizes
        self.target_sizes = target_sizes
        self.n_lvls = len(feat_strides)
        assert len(kernel_sizes)==1 and len(target_sizes)==1, (len(kernel_sizes), len(target_sizes))

    def _get_kernel_crop_modules(self, in_channels, kernel_size, spatial_scale):
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
        return kernel_crop_module.cuda()

    def _get_target_crop_modules(self, in_channels, kernel_size, spatial_scale):
        target_crop_module = RoIAlign(kernel_size, spatial_scale)
        return target_crop_module.cuda()

    def _adjust_rpn_center(self, roi_center_x, roi_center_y, radius, min_w, max_w, min_h, max_h):
        tensor_zero = roi_center_x.new_zeros(1)
        roi_center_x = roi_center_x - torch.min(roi_center_x - radius - min_w, tensor_zero)
        roi_center_x = roi_center_x - torch.max(roi_center_x + radius - max_w, tensor_zero)
        roi_center_y = roi_center_y - torch.min(roi_center_y - radius - min_h, tensor_zero)
        roi_center_y = roi_center_y - torch.max(roi_center_y + radius - max_h, tensor_zero)
        return roi_center_x, roi_center_y

    def _pad_feature_according_to_rois(self, features, feat_stride, rois):
        x1 = rois[:, 1]
        y1 = rois[:, 2]
        x2 = rois[:, 3]
        y2 = rois[:, 4]
        x1 = torch.min(x1)
        y1 = torch.min(y1)
        x2 = torch.max(x2)
        y2 = torch.max(y2)
        pad_left, pad_right, pad_up, pad_down = [0, 0, 0, 0]
        h, w = features.size()[2:]
        h = h*feat_stride
        w = w*feat_stride
        extra = 0 - x1
        if extra > 0:
            pad_left = torch.ceil(extra / feat_stride).type(torch.int).item()
        extra = 0 - y1
        if extra > 0:
            pad_up = torch.ceil(extra / feat_stride).type(torch.int).item()
        extra = x2-(w-1)*feat_stride
        if extra > 0:
            pad_right = torch.ceil(extra / feat_stride).type(torch.int).item()
        extra = y2-(h-1)*feat_stride
        if extra > 0:
            pad_down = torch.ceil(extra / feat_stride).type(torch.int).item()

        features = F.pad(features, pad=(pad_left, pad_right, pad_up, pad_down), mode='constant', value=0)
        rois[:,1:] = rois[:,1:]+rois.new_tensor([pad_left, pad_up, pad_left, pad_up])*feat_stride
        return features, rois

    def _get_kernels_targets_single_lvl(self, feat1, feat2, img_meta, feat_stride, kernel_size, target_size, rpn_rois,
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
        r = (kernel_size - 1) / 2.0 * feat_stride

        roi_kernels[:, 1] = rpn_rois_center_x - r
        roi_kernels[:, 3] = rpn_rois_center_x + r
        roi_kernels[:, 2] = rpn_rois_center_y - r
        roi_kernels[:, 4] = rpn_rois_center_y + r

        # targets
        r = (target_size - 1) / 2.0 * feat_stride
        #rpn_rois_center_x, rpn_rois_center_y = self._adjust_rpn_center(rpn_rois_center_x, rpn_rois_center_y, r, 0,
        #                                                               feat_stride*(w-1), 0, feat_stride*(h-1))
        roi_targets[:, 1] = rpn_rois_center_x - r
        roi_targets[:, 3] = rpn_rois_center_x + r
        roi_targets[:, 2] = rpn_rois_center_y - r
        roi_targets[:, 4] = rpn_rois_center_y + r
        #feat1_padded, roi_kernels = self._pad_feature_according_to_rois(feat1, feat_stride, roi_kernels)
        #feat2_padded, roi_targets = self._pad_feature_according_to_rois(feat2, feat_stride, roi_targets)
        kernels = kernel_crop_module(feat1, roi_kernels)
        targets = target_crop_module(feat2, roi_targets)
        target_ranges = roi_targets
        target_metas = [img_meta[int(roi[0].item())] for roi in rpn_rois]

        return kernels, targets, target_ranges, target_metas

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
        # if padding, rois should shift.
        #feat1_padded, roi_kernels = self._pad_feature_according_to_rois(feat1, feat_stride, roi_kernels)
        #feat2_padded, roi_targets = self._pad_feature_according_to_rois(feat2, feat_stride, roi_targets)

        kernels = kernel_crop_module(feat1, roi_kernels)
        targets = target_crop_module(feat2, roi_targets)
        target_ranges = roi_targets
        target_metas = [img_meta[int(roi[0].item())] for roi in rpn_rois]

        return kernels, targets, target_ranges, target_metas


    def forward_single(self, feat1, feat2, img_meta, feat_stride, kernel_size, target_size, rpn_rois_1, kernel_crop_module,
                                                                               target_crop_module):
        kernels, targets, target_ranges, target_metas = self._get_kernels_targets_single_lvl(feat1, feat2, img_meta, feat_stride,
                                                                               kernel_size, target_size,
                                                                               rpn_rois_1, kernel_crop_module,
                                                                               target_crop_module)
        cls_score, bbox_pred = self.rpn_module(kernels, targets)
        return cls_score, bbox_pred, target_ranges, target_metas

    def forward(self, feat1s, feat2s, rpn_rois_1, img_metas):
        return multi_apply(self.forward_single, feat1s, feat2s, [img_metas]*self.n_lvls, self.feat_strides, self.kernel_sizes, self.target_sizes,
                           [rpn_rois_1]*self.n_lvls, self.kernel_crop_modules, self.target_crop_modules)

    def loss(self,
             cls_scores,
             bbox_preds,
             target_ranges,
             img_metas,
             gt_bboxes,
             cfg,
             gt_bboxes_ignore=None):
        # img_metas are the same for all levels, use img_metas[0] instead.
        assert len(target_ranges)==1, 'Only support 1 level feature map.'
        target_ranges_for_batches = target_ranges[0]
        gt_bboxes_shifted = [(gt_b - tgt_rng[1:3].repeat(2)) if gt_b is not None else gt_b for gt_b, tgt_rng in \
                             zip(gt_bboxes, target_ranges_for_batches)]

        #scales_x = (target_ranges_for_batches[:, 3] - target_ranges_for_batches[:, 1]) / ((self.target_sizes[0]-1)*self.feat_strides[0])
        #scales_y = (target_ranges_for_batches[:, 4] - target_ranges_for_batches[:, 2]) / ((self.target_sizes[0]-1)*self.feat_strides[0])

        #gt_bboxes_shifted = [(gt_b-tgt_rng[1:3].repeat(2))/gt_b.new_tensor([s_x, s_y, s_x, s_y]) \
        #                     if gt_b is not None else gt_b for gt_b, tgt_rng, s_x, s_y in \
        #                     zip(gt_bboxes, target_ranges_for_batches, scales_x, scales_y)]

        losses = super(SiameseRPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes_shifted,
            None,
            img_metas[0],
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_siamese_rpn_cls=losses['loss_cls'], loss_siamese_rpn_bbox=losses['loss_bbox'])

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, target_ranges, img_metas, rpn_rois_1, cfg,
                   rescale=False):
        # img_metas: [lvls][batches][meta], different lvls have same metas.
        assert len(cls_scores) == len(bbox_preds)
        assert len(rpn_rois_1) == len(img_metas[0])
        num_levels = len(cls_scores)
        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],
                                                   self.anchor_strides[i])
            for i in range(num_levels)
        ]

        result_list = [[] for _ in range(64)] # if 64 is small, increase it.
        for img_id in range(len(img_metas[0])):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            target_ranges_list = [
                target_ranges[i][img_id].detach() for i in range(num_levels)
            ]
            rpn_roi = rpn_rois_1[img_id]

            img_shape = img_metas[0][img_id]['img_shape']
            scale_factor = img_metas[0][img_id]['scale_factor']
            proposals = self.get_bboxes_single(rpn_roi, cls_score_list, bbox_pred_list, target_ranges_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, cfg, rescale)

            result_list[int(target_ranges_list[0][0].item())].append(proposals)

        for idx, proposals in enumerate(result_list):
            if len(proposals)>0:
                result_list[idx] = torch.cat(proposals, dim=0)
        return result_list

    def get_bboxes_single(self,
                          rpn_roi,
                          cls_scores,
                          bbox_preds,
                          target_ranges,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        mlvl_scores = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, max_shape=None)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            # shift proposal accordingly.

            #s_x = (target_ranges[idx][3] - target_ranges[idx][1]) / ((self.target_sizes[idx]-1)*self.feat_strides[idx])
            #s_y = (target_ranges[idx][4] - target_ranges[idx][2]) / ((self.target_sizes[idx]-1)*self.feat_strides[idx])
            #proposals = proposals*proposals.new_tensor([s_x, s_y, s_x, s_y]) + target_ranges[idx][1:3].repeat(2)
            proposals = proposals + target_ranges[idx][1:3].repeat(2)

            proposals[:, 0] = proposals[:, 0].clamp(min=0, max=img_shape[1] - 1)
            proposals[:, 1] = proposals[:, 1].clamp(min=0, max=img_shape[0] - 1)
            proposals[:, 2] = proposals[:, 2].clamp(min=0, max=img_shape[1] - 1)
            proposals[:, 3] = proposals[:, 3].clamp(min=0, max=img_shape[0] - 1)
            scores = scores.unsqueeze(-1)
            proposals = torch.cat([proposals, scores], dim=-1)
            tracking_scores = self.nms_tracking_scores(proposals, rpn_roi, cfg)
            proposals[:, -1] = tracking_scores
            proposals, inds = nms(proposals, cfg.nms_thr)
            scores = scores[inds,:]
            proposals = proposals[:cfg.nms_post, :]
            scores = scores[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
            mlvl_scores.append(scores)
        proposals = torch.cat(mlvl_proposals, 0)
        proposal_scores = torch.cat(mlvl_scores, 0)
        if cfg.nms_across_levels:
            proposals, inds = nms(proposals, cfg.nms_thr)
            proposal_scores = proposal_scores[inds,:]
            proposals = proposals[:cfg.max_num, :]
            proposal_scores = proposal_scores[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
            proposal_scores = proposal_scores[topk_inds, :]

        assert len(proposals)==len(proposal_scores)
        proposals[:, -1] = proposal_scores[:, 0]
        return proposals

    def change(self, r):
        return torch.max(r, 1. / r)

    def sz(self,w,h):
        pad = (w+h)*0.5
        return torch.sqrt((w+pad)*(h+pad))

    def nms_tracking_scores(self, proposals, rpn_roi_track, cfg):
        bw = rpn_roi_track[3] - rpn_roi_track[1]
        bh = rpn_roi_track[4] - rpn_roi_track[2]
        bx = (rpn_roi_track[3] + rpn_roi_track[1])/2.0
        by = (rpn_roi_track[4] + rpn_roi_track[2])/2.0
        pw = proposals[:, 2] - proposals[:, 0]
        ph = proposals[:, 3] - proposals[:, 1]
        px = (proposals[:, 2] + proposals[:, 0])/2.0
        py = (proposals[:, 3] + proposals[:, 1])/2.0
        score = proposals[:, 4]
        # aspect ratio penalty.
        r_c = self.change((pw/ph)/(bw/bh))
        # scale penalty.
        s_c = self.change(self.sz(pw, ph)/self.sz(bw, bh))
        penalty = torch.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score
        # window penalty.
        distance = ((px-bx).pow(2)+(py-by).pow(2)).sqrt()
        window_sz = self.sz(bw, bh)/2.0
        windows_penalty = 0.5+0.5*((distance*3.141592653589793/window_sz).cos())
        zero_inds = torch.nonzero((distance > window_sz).view(-1))
        if zero_inds.size(0) > 0:
            windows_penalty[zero_inds[:, 0]] = 0.
        pscore = pscore*(1 - cfg.TRACK.WINDOW_INFLUENCE) + windows_penalty * cfg.TRACK.WINDOW_INFLUENCE
        return pscore
