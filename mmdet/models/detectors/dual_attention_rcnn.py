from ..registry import DETECTORS
from .two_stage import TwoStageDetector
from .. import builder

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.ops import nms
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, auto_fp16, bbox2delta, delta2bbox, \
    bbox_overlaps,multiclass_nms, multiclass
from ..utils import ConvModule
from ...datasets.transforms import BboxTransform
import random
import numpy as np
from copy import deepcopy
from mmdet import ops
from collections import deque as Deque

class KeyValueEncoder(nn.Module):
    # Using key branch for encoding feature maps.
    def __init__(self, in_channels, dim_key, dim_val):
        super(KeyValueEncoder, self).__init__()
        self.key = nn.Conv2d(in_channels, dim_key, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv2d(in_channels, dim_val, kernel_size=1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, logits):
        k = self.key(logits)
        v = self.value(logits)
        return k, v

class ChannelAttetionPropagation1D(nn.Module):
    def __init__(self, ):
        super(ChannelAttetionPropagation1D, self).__init__()
        self.alpha = nn.Parameter(nn.Tensor(1).fill_(0), requires_grad=True)

    def forward(self, key_mem, val_mem, key_cur, val_cur):
        '''
        :param key_mem: (N,t,n,c).
        :param key_cur: (N,n,c).
        :return: (N,n,c).
        '''
        N,n,C = key_cur.shape
        _,T,n,C2 = val_mem.shape
        key_cur = key_cur # (N,n,C)
        key_mem = key_mem.permute(0,3,1,2).view(N, C, -1) #(N,C,t*n)
        val_mem = val_mem.view(N, -1, C2) # (N,t*n,c2)
        kv_mul = torch.matmul(key_mem, val_mem) # /((n*T)) # TODO check need any normalization like sqrt(len)?
        kv_mul = F.softmax(kv_mul, dim=1)
        out = torch.matmul(key_cur, kv_mul)
        out = self.alpha*out + val_cur
        return out

class SpatialAttentionPropagation1D(nn.Module):
    def __init__(self, ):
        super(ChannelAttetionPropagation1D, self).__init__()
        self.alpha = nn.Parameter(nn.Tensor(1).fill_(0), requires_grad=True)

    def forward(self, key_mem, val_mem, key_cur, val_cur):
        '''
        :param key_mem: (N,t,n,c).
        :param key_cur: (N,n,c).
        :return: (N,n,c).
        '''
        N,n,C = key_cur.shape
        _,T,n,C2 = val_mem.shape
        key_cur = key_cur  # (N,n,C)
        key_mem = key_mem.permute(0, 3, 1, 2).view(N, C, -1)  # (N,C,t*n)
        kk_mul = torch.matmul(key_cur, key_mem) #(N,n,t*n)
        kk_mul = F.softmax(kk_mul, dim=-1)
        val_mem = val_mem.view(N, -1, C2)  # (N,t*n,c2)
        out = torch.matmul(kk_mul, val_mem) # (N,n,c2)
        out = self.alpha*out + val_cur
        return out

class MemoryContainer1D(nn.Module):
    def __init__(self, max_size):
        super(MemoryContainer1D, self).__init__()
        self.mem_queue = Deque()
        assert max_size>0, 'Memory container should have max size larger than 0.'
        self.max_size = max_size

    def is_empty(self):
        return len(self.mem_queue)==0

    def is_full(self):
        return len(self.mem_queue)==self.max_size

    def size(self):
        return len(self.mem_queue)

    def add_memory(self, logits):
        assert logits.ndim==3, 'input memory should have shape of (N,n,C)'
        while len(self.mem_queue)>=self.max_size:
            self.mem_queue.popleft()
        self.mem_queue.append(logits)

    def get_stacked_memory(self):
        return torch.stack(list(self.mem_queue), dim=1) # shape of (N,T,n,C)

class BoxFeatureConv(nn.Module):
    def __init__(self, roi_feat_size, num_convs, num_fcs, in_channels, fc_out_channels,conv_cfg=None, norm_cfg=None,):
        super(BoxFeatureConv, self).__init__()
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.in_channels = in_channels
        self.fc_out_channels = fc_out_channels
        self.roi_feat_size = roi_feat_size
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.convs, self.fcs, last_layer_dim = \
            self._add_conv_fc_branch(self.num_convs, self.num_fcs, self.in_channels, True)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        if self.num_shared_fcs > 0:
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        return x

@DETECTORS.register_module
class DualAttentionRCNN(TwoStageDetector, '''??testmixin'''):
    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 dual_attention_cfg,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None,
                 freeze_feature_extractor=False,
                 freeze_backbone=True,
                 train_rcnn=True,
                 T=1):
        super(DualAttentionRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.da_cfg = dual_attention_cfg
        self.kv_enc = KeyValueEncoder(self.da_cfg.in_channels, self.da_cfg.dim_key, self.da_cfg.dim_val)
        self.ch_att = ChannelAttetionPropagation1D(self.da_cfg.key_mem, self.da_cfg.val_mem, self.da_cfg.key_cur, self.da_cfg.val_cur)
        self.sp_att = SpatialAttentionPropagation1D(self.da_cfg.key_mem, self.da_cfg.val_mem, self.da_cfg.key_cur, self.da_cfg.val_cur)
        self.box_feat_conv = BoxFeatureConv(
            self.da_cfg.roi_feat_size,
            self.da_cfg.num_box_feat_convs,
            self.da_cfg.num_box_feat_fcs,
            self.da_cfg.num_box_feat_in_channels,
            self.da_cfg.fc_out_channels,
            self.da_cfg.conv_cfg,
            self.da_cfg.norm_cfg,)

    def forward(self, imgs, img_meta, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, proposals=None):
        N = imgs.shape[0]
        H = imgs.shape[3]
        W = imgs.shape[4]
        T = self.T
        tgt_t = int((T - 1) / 2)
        imgs = imgs.view(N * T, 3, H, W)
        extracted_feat = self.extract_feat(imgs)
        losses = dict()
        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(extracted_feat)
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)

            rpn_outs_tgt = self.rpn_head(extracted_feat[N * tgt_t:N * (tgt_t + 1)])
            proposal_inputs_tgt = rpn_outs_tgt + (img_meta, proposal_cfg)
            proposal_list_tgt = self.rpn_head.get_bboxes(*proposal_inputs_tgt)

            if losses is not None:
                # TODO test rpn outs.
                rpn_loss_inputs = rpn_outs[N*tgt_t:N*(tgt_t+1)] + (gt_bboxes, img_meta, self.train_cfg.rpn)
                rpn_losses = self.rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
                losses.update(rpn_losses)

        # assign gts and sample proposals
        bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
        bbox_sampler = build_sampler(self.train_cfg.rcnn.sampler, context=self)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(N)]
        sampling_results = []
        for i in range(N):
            if gt_bboxes[i] is None or len(gt_bboxes[i]) == 0:
                continue
            assign_result = bbox_assigner.assign(proposal_list_tgt[i],
                                                 gt_bboxes[i],
                                                 gt_bboxes_ignore[i],
                                                 gt_labels[i])
            sampling_result = bbox_sampler.sample(assign_result,
                                                  proposal_list_tgt[i],
                                                  gt_bboxes[i],
                                                  gt_labels[i],
                                                  feats=[lvl_feat[i][None] for lvl_feat in extracted_feat])
            sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi(proposal_list)
            bbox_feats = self.bbox_roi_extractor(extracted_feat[:self.bbox_roi_extractor.num_inputs], rois)
            # TODO: a more flexible way to decide which feature maps to use
            rois_tgt = bbox2roi([res.bboxes for res in sampling_results])
            print(bbox_feats.shape)
            bbox_feats_tgt = self.bbox_roi_extractor(extracted_feat[N * tgt_t:N * (tgt_t + 1)][:self.bbox_roi_extractor.num_inputs], rois_tgt)
            bbox_feats = self.box_feat_conv(bbox_feats)
            bbox_feats_tgt = self.box_feat_conv(bbox_feats_tgt)
            ref_keys, ref_vals = self.kv_enc(bbox_feats)
            cur_key, cur_val = self.kv_enc(bbox_feats_tgt)
            cur_val = self.sp_att(ref_keys, ref_vals, cur_key, cur_val)
            cur_val = self.ch_att(ref_keys, ref_vals, cur_key, cur_val)
            cls_score, bbox_pred = self.bbox_head(cur_val)
            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
            losses.update(loss_bbox)

        return losses

    # TODO modify the test.
    def simple_test_vid(self, imgs, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        # imgs: N,T,C,H,W.
        N = imgs.shape[0]
        H = imgs.shape[3]
        W = imgs.shape[4]
        T = self.T
        imgs = imgs.view(N * T, 3, H, W)
        feats = self.extract_feat(imgs)[0]
        feats = feats.view(N, T, -1, feats.shape[-2], feats.shape[-1])
        tgt_t = int((T - 1) / 2)
        tgt_feature = feats[:, tgt_t, :, :, :].contiguous()

        proposal_list = self.simple_test_rpn(
            [tgt_feature], img_meta, self.test_cfg.rpn) if proposals is None else proposals
        rpn_rois_1 = bbox2roi([prop[:, :4] for prop in proposal_list])

        ########
        block_feats = self.track_block_feat_sequential(tgt_feature, feats, rpn_rois_1, img_meta, T, tgt_t)
        #block_feats = self.track_block_feat_instant(tgt_feature, feats, rpn_rois_1, img_meta, T, tgt_t)

        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        cls_score, bbox_pred = self.bbox_head(block_feats)

        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rpn_rois_1,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=self.test_cfg.rcnn)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                block_feats, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results