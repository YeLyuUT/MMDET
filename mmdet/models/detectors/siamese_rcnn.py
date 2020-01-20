from ..registry import DETECTORS
from .two_stage import TwoStageDetector
from .. import builder

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.ops import nms
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, auto_fp16, bbox2delta, delta2bbox, \
    bbox_overlaps,multiclass_nms, multiclass
from .test_mixins import SiameseRPNTestMixin
from ...datasets.transforms import BboxTransform
import random
import numpy as np
from copy import deepcopy
from mmdet import ops

import time
class clock():
    def tic(self):
        self._tic = time.time()
    def toc(self,text=''):
        self._toc = time.time()
        print(text,':', self._toc-self._tic)
        self._tic = time.time()
Timer = clock()

class _inner_block(nn.Module):
    def __init__(self, inplanes, planes):
        super(_inner_block, self).__init__()
        self.conv1 = self.conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.normal_init(self.conv1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        self.normal_init(self.conv2, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def conv3x3(self, inplanes, planes):
        return nn.Conv2d(inplanes, planes, 3, 1, 1)

    def forward(self, x):
        residual = x
        # normal convolutions.
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

@DETECTORS.register_module
class SiameseRCNN(TwoStageDetector, SiameseRPNTestMixin):
    def __init__(self,
                 siameserpn_head,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None,
                 img_train=False,
                 vid_train=False,
                 track_train=False,
                 graphnn_train=False,
                 freeze_feature_extractor=False,
                 freeze_backbone=False,
                 train_rcnn=True,
                 detach_track_feature=False,
                 T=1,
                 space_time_augmentation=None):
        super(SiameseRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        assert siameserpn_head is not None
        self.siameserpn_head = builder.build_head(siameserpn_head)
        self.rois_tracked = None
        self.extracted_feat1 = None
        self.bbox_transform = BboxTransform()
        self.img_train_method = 1 #0 for both, 1 for left, 2 for right.
        self.img_train = img_train
        self.vid_train = vid_train
        self.track_train = track_train
        self.graphnn_train = graphnn_train
        assert not (self.img_train and self.vid_train)
        self.freeze_feature_extractor = freeze_feature_extractor
        self.freeze_backbone = freeze_backbone
        self.train_rcnn = train_rcnn
        if self.freeze_feature_extractor:
            self.freeze_parts(self.backbone)
            self.freeze_parts(self.neck)
        if self.freeze_backbone:
            self.freeze_parts(self.backbone)
        self.T = T
        self.sequence_mapped_bboxes = None
        self.sequence_mapped_bboxes_result = None
        self.sequence_buffer = None
        self.sequence_buffer_length = 1
        self.sequence_gap = 1
        self.multi_track_max_gap = 10
        self.sequence_counter = 0
        self._proposal_repo = None
        self.detach_track_feature = detach_track_feature
        #self.nms = trNMS(cfg.SIAMESE.PANELTY_K, cfg.SIAMESE.HANNING_WINDOW_WEIGHT, cfg.SIAMESE.HANNING_WINDOW_SIZE_FACTOR)
        self.space_time_augmentation = space_time_augmentation
        if self.space_time_augmentation is not None:
            nlvls = self.space_time_augmentation.levels
            C_in = self.space_time_augmentation.C_in
            C_qk = self.space_time_augmentation.C_qk
            relation_percent = self.space_time_augmentation.relation_percent
            print('relation_percent set to:', relation_percent)
            self.space_time_modules = [ops.PointwiseGraphNN(C_in, C_qk=C_qk, relation_percent=relation_percent).cuda() for _ in range(nlvls)]
            self.space_time_mem = []
            self.space_time_mem_counter = 0
        else:
            self.space_time_modules = None

    @auto_fp16(apply_to=('img',))
    def forward(self, return_loss = True, **inputs):
        if return_loss is True:
            if self.img_train:
                return self.forward_img_train(**inputs)
            elif self.vid_train:
                return self.forward_train_vid(**inputs)
            elif self.track_train:
                return self.forward_train_track_only(**inputs)
            elif self.graphnn_train:
                return self.forward_train_graphNN(**inputs)
            else:
                return self.forward_train(**inputs)
        else:
            return self.forward_test(inputs.pop('img', None), inputs.pop('img_meta', None), **inputs)

    def freeze_parts(self, network):
        print('Freezing parameters:{}'.format(network.parameters()))
        for param in network.parameters():
            param.requires_grad = False

    def unfreeze_parts(self, network):
        print('UnFreezing parameters:{}'.format(network.parameters()))
        for param in network.parameters():
            param.requires_grad = True

    def proposals_repo(self):
        if self._proposal_repo is None:
            self._proposal_repo = []
        return self._proposal_repo

    def append_to_proposals_repo(self, proposals):
        self.proposal_repo().append(proposals)
        return True

    def get_last_proposals(self):
        if len(self.proposal_repo())==0:
            return None
        else:
            return self.proposal_repo()[-1]

    def _get_mapper_gt1_to_gt2(self, gt_bboxes_1, gt_ids_1, gt_bboxes_2, gt_ids_2):
        mappers=[dict() for _ in range(len(gt_bboxes_1))]
        for img_id in range(len(mappers)):
            box2s = gt_bboxes_2[img_id]
            id2s = gt_ids_2[img_id]
            for ind, id2 in enumerate(id2s):
                mapped_gt2_box = box2s[ind:ind+1, :]
                mappers[img_id][id2.item()] = mapped_gt2_box
        return mappers

    def random_boxes_from_gts(self, gt_boxes, n_samples_per_gt):
        deltas = gt_boxes.new_zeros(len(gt_boxes), n_samples_per_gt, 4).uniform_(-1., 1.)
        # deltas [[[dx,dy,dw,dh]]]
        deltas[:, :, 2:] = (deltas[:, :, 2:] + 1) / 2.0 + 0.5
        dx, dy, dw, dh = (deltas[:,:,0:1], deltas[:,:,1:2], deltas[:,:,2:3], deltas[:,:,3:4])
        x1, y1, x2, y2 = (gt_boxes[:,0:1], gt_boxes[:,1:2], gt_boxes[:,2:3], gt_boxes[:,3:4])
        w = x2 - x1
        h = y2 - y1
        x = (x1 + x2) / 2.0
        y = (y1 + y2) / 2.0
        new_x = dx * w[:,None,:] + x[:,None,:]
        new_y = dy * h[:,None,:] + y[:,None,:]
        new_w = dw * w[:,None,:] + 1e-5
        new_h = dh * h[:,None,:] + 1e-5
        new_x1, new_y1, new_x2, new_y2 = (new_x - new_w / 2.0, new_y - new_h / 2.0, new_x + new_w / 2.0, new_y + new_h / 2.0)
        proposals = torch.cat([new_x1, new_y1, new_x2, new_y2], dim=-1).reshape(-1, 4)
        return proposals

    def merge_features(self, feature_list):
        num_levels = len(feature_list)
        refine_level = int((num_levels - 1) / 2)
        out_size = feature_list[refine_level].size()[2:]
        out_feature_list = []
        for i in range(num_levels):
            if i != refine_level:
                feat = F.interpolate(feature_list[i], size=out_size, mode='bilinear')
            else:
                feat = feature_list[i]
            out_feature_list.append(feat)
        out_features = [torch.cat(out_feature_list, dim=1)]
        return out_features

    def forward_rpn(self,
                    extracted_feat,
                    img_meta,
                    gt_bboxes,
                    gt_bboxes_ignore=None,
                    losses=None):
        rpn_outs = self.rpn_head(extracted_feat)
        rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                      self.train_cfg.rpn)

        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                          self.test_cfg.rpn)

        proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)

        if losses is not None:
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)
        return proposal_list, losses

    def forward_rcnn_train(self,
                     num_imgs,
                     extracted_feat,
                     proposal_list,
                     gt_bboxes,
                     gt_labels,
                     gt_bboxes_ignore=None,
                     gt_masks=None,
                     losses=None):
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                if gt_bboxes[i] is None or len(gt_bboxes[i]) == 0:
                    continue
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in extracted_feat])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                extracted_feat[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    extracted_feat[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(sampling_results,
                                                     gt_masks,
                                                     self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)
        return losses

    def forward_track_train(self,
                            n_batches,
                            proposal_list_1,
                            extracted_features_1,
                            extracted_features_2,
                            gt_bboxes1,
                            gt_bboxes2,
                            gt_trackids1,
                            gt_trackids2,
                            img_meta,
                            ):
        if len(extracted_features_1)>1:
            extracted_features_1 = extracted_features_1[1:-1]
            extracted_features_2 = extracted_features_2[1:-1]
            extracted_features_1 = self.merge_features(extracted_features_1)
            extracted_features_2 = self.merge_features(extracted_features_2)

        if self.detach_track_feature:
            extracted_features_1 = extracted_features_1.detach()
            extracted_features_2 = extracted_features_2.detach()
        # Get rpn proposals.
        # proposal_list_1 = proposal_list[:n_batches]
        # proposal_list_2 = proposal_list[n_batches:]
        # Get training rois.
        tracking_bbox_assigner = build_assigner(self.train_cfg.siameserpn.assigner_track)
        tracking_bbox_sampler = build_sampler(self.train_cfg.siameserpn.sampler_track, context=self)
        sampling_results = []
        for i in range(n_batches):
            proposal_list = self.random_boxes_from_gts(gt_bboxes1[i], 256)
            #proposal_list = proposal_list_1[i]
            assign_result_1 = tracking_bbox_assigner.assign(proposal_list,
                                                            gt_bboxes1[i],
                                                            None,
                                                            gt_trackids1[i])
            sampling_result_1 = tracking_bbox_sampler.sample(assign_result_1,
                                                             proposal_list,
                                                             gt_bboxes1[i],
                                                             gt_trackids1[i])

            sampling_results.append(sampling_result_1)

        # Only pos boxes are used.
        rpn_rois_1 = bbox2roi([res.pos_bboxes for res in sampling_results])
        mapper_gt1_to_gt2 = self._get_mapper_gt1_to_gt2(gt_bboxes1, gt_trackids1, gt_bboxes2, gt_trackids2)
        siameserpn_gt_boxes = []
        siameserpn_gt_ids = []
        for i in range(n_batches):
            res = sampling_results[i]
            lbls = res.pos_gt_labels
            for idx in range(len(lbls)):
                lbl = lbls[idx]
                lbl_val = lbl.item()
                if lbl_val in mapper_gt1_to_gt2[i].keys():
                    mapped_gt2_box = mapper_gt1_to_gt2[i][lbl_val]
                else:
                    mapped_gt2_box = None
                siameserpn_gt_boxes.append(mapped_gt2_box)
                if mapped_gt2_box is not None:
                    siameserpn_gt_ids.append(lbl)
                else:
                    siameserpn_gt_ids.append(None)

        gt_bboxes = siameserpn_gt_boxes
        gt_ids = siameserpn_gt_ids

        cls_score, bbox_pred, target_ranges, target_metas = \
            self.siameserpn_head(extracted_features_1, extracted_features_2, rpn_rois_1, img_meta)
        assert len(cls_score) == 1 and len(bbox_pred) == 1
        cls_score = cls_score[0]
        bbox_pred = bbox_pred[0]

        bbox_targets = self.siameserpn_head.get_target(rpn_rois_1,
                                                       gt_bboxes, gt_ids,
                                                       self.train_cfg.rcnn)
        siameserpn_losses, pred_bboxes = self.siameserpn_head.loss(rpn_rois_1, cls_score, bbox_pred, *bbox_targets)
        proposal_list = []
        for idx in range(n_batches):
            inds = rpn_rois_1[:, 0]==idx
            proposal_list.append(pred_bboxes[inds])

        return siameserpn_losses, proposal_list

    def forward_img_train(self, img, img_meta, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, proposals=None):
        n_batches = img.shape[0]
        x = self.extract_feat(img)

        losses = dict()
        # RPN forward and loss
        if self.with_rpn:
            proposal_list, losses = self.forward_rpn(x, img_meta, gt_bboxes, gt_bboxes_ignore, losses)
        else:
            proposal_list = proposals
        if self.train_rcnn:
            losses = self.forward_rcnn_train(n_batches,
                                             x,
                                             proposal_list,
                                             gt_bboxes,
                                             gt_labels,
                                             gt_bboxes_ignore,
                                             gt_masks,
                                             losses)
        return losses

    def track_block_feat_sequential(self, tgt_feature, feats, rpn_rois_1, img_meta, T, tgt_t):
        xt_list = []
        for t in range(T):
            xt = feats[:, t, :, :, :].contiguous()
            xt_list.append(xt)
        bbox_feats_list = []
        # backward track.
        tmp_track_feat = [tgt_feature]
        tmp_track_rois = rpn_rois_1
        for t in range(tgt_t):
            feat2 = [xt_list[tgt_t - 1 - t]]
            _proposal_list = self.siamese_rpn(tmp_track_feat, feat2, tmp_track_rois, img_meta)
            # TODO check here.
            rpn_rois_2 = bbox2roi([prop[:, :4] for prop in _proposal_list])
            bbox_feats = self.bbox_roi_extractor(feat2, rpn_rois_2)
            bbox_feats_list = [bbox_feats] + bbox_feats_list
            tmp_track_feat = feat2
            assert rpn_rois_2.shape == tmp_track_rois.shape, (rpn_rois_2.shape, tmp_track_rois.shape)
            tmp_track_rois = rpn_rois_2
        # add middle features.
        bbox_feats_list.append(self.bbox_roi_extractor([tgt_feature], rpn_rois_1))
        # forward track.
        tmp_track_feat = [tgt_feature]
        tmp_track_rois = rpn_rois_1
        for t in range(tgt_t + 1, T):
            feat2 = [xt_list[t]]
            _proposal_list = self.siamese_rpn(tmp_track_feat, feat2, tmp_track_rois, img_meta)
            # TODO check here.
            rpn_rois_2 = bbox2roi([prop[:, :4] for prop in _proposal_list])
            bbox_feats = self.bbox_roi_extractor(feat2, rpn_rois_2)
            bbox_feats_list = bbox_feats_list + [bbox_feats]
            tmp_track_feat = feat2
            assert rpn_rois_2.shape == tmp_track_rois.shape
            tmp_track_rois = rpn_rois_2

        block_feats = torch.cat(bbox_feats_list, dim=1)
        return block_feats

    '''
    def track_block_feat_instant(self, tgt_feature, feats, rpn_rois_1, img_meta, T, tgt_t):
        # backward track.
        #feat2_list, tgt_feature_list = zip(*[(feats[:,t,:,:,:], tgt_feature) for t in range(T) if t!=tgt_t])
        #feat2s = torch.cat(feat2_list, dim=0)
        #tgt_feats = torch.cat(tgt_feature_list, axis=0)
        feat2s = torch.cat([feats[:,:tgt_t,:,:,:], feats[:,tgt_t+1:,:,:,:]], dim=1)
        tgt_feats = torch.stack([tgt_feature]*(T-1), dim=1)
        for i in range(T-1):
            rpn_rois_1_tmp = rpn_rois_1.clone()
            rpn_rois_1_tmp[:,0]=rpn_rois_1_tmp[:,0]+i*tgt_feature.shape[0]???N,T, rois
        assert feat2s.shape==tgt_feats.shape, (feat2s.shape, tgt_feats.shape)
        feat2s = feat2s.view(-1,feat2s.shape[-3],feat2s.shape[-2],feat2s.shape[-1])
        tgt_feats = tgt_feats.view(-1, tgt_feats.shape[-3],tgt_feats.shape[-2],tgt_feats.shape[-1])
        _proposal_list = self.siamese_rpn([tgt_feats], [feat2s], , img_meta)
        # TODO check here.
        rpn_rois_2 = bbox2roi([prop[:, :4] for prop in _proposal_list])
        bbox_feats = self.bbox_roi_extractor(feat2s, rpn_rois_2)
        bbox_feats = bbox_feats.view(tgt_feature.shape[0], -1, bbox_feats.shape[-2], bbox_feats.shape[-1])
        tgt_bbox_feat = self.bbox_roi_extractor([tgt_feature], rpn_rois_1)
        bbox_feats_list = [tgt_bbox_feat, bbox_feats]
        block_feats = torch.cat(bbox_feats_list, dim=1)
        return block_feats
    '''

    def forward_train_vid(self, imgs, img_meta, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, proposals=None):
        #imgs: N,T,C,H,W.
        N = imgs.shape[0]
        H = imgs.shape[3]
        W = imgs.shape[4]
        T = self.T
        imgs = imgs.view(N*T,3,H,W)
        feats = self.extract_feat(imgs)[0]
        feats = feats.view(N,T,-1,feats.shape[-2],feats.shape[-1])
        tgt_t = int((T-1)/2)
        tgt_feature = feats[:, tgt_t, :, :, :].contiguous()

        losses = dict()
        # RPN forward and loss
        if self.with_rpn:
            proposal_list, losses = self.forward_rpn([tgt_feature], img_meta, gt_bboxes, gt_bboxes_ignore, losses)

        # assign gts and sample proposals
        bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
        bbox_sampler = build_sampler(
            self.train_cfg.rcnn.sampler, context=self)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(N)]
        sampling_results = []
        for i in range(N):
            if gt_bboxes[i] is None or len(gt_bboxes[i]) == 0:
                continue
            assign_result = bbox_assigner.assign(proposal_list[i],
                                                 gt_bboxes[i],
                                                 gt_bboxes_ignore[i],
                                                 gt_labels[i])
            sampling_result = bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in [tgt_feature]])
            sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rpn_rois_1 = bbox2roi([res.bboxes for res in sampling_results])

        ########
        block_feats = self.track_block_feat_sequential(tgt_feature, feats, rpn_rois_1, img_meta, T, tgt_t)
        #block_feats = self.track_block_feat_instant(tgt_feature, feats, rpn_rois_1, img_meta, T, tgt_t)

        if self.with_shared_head:
            block_feats = self.shared_head(block_feats)
        cls_score, bbox_pred = self.bbox_head(block_feats)

        bbox_targets = self.bbox_head.get_target(sampling_results,
                                                 gt_bboxes, gt_labels,
                                                 self.train_cfg.rcnn)
        loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                        *bbox_targets)

        losses.update(loss_bbox)
        return losses

    def forward_train_track_only(self,
                                img1,
                                img2,
                                img_meta,
                                gt_bboxes1,
                                gt_bboxes2,
                                gt_labels1,
                                gt_labels2,
                                gt_trackids1,
                                gt_trackids2,
                                gt_bboxes_ignore1=None,
                                gt_bboxes_ignore2=None,
                                gt_masks1=None,
                                gt_masks2=None,
                                proposals=None):
        ##################################
        #      Detection RPN part        #
        ##################################
        # same as two stage detector
        img = torch.cat([img1, img2], dim=0)
        n_batches = img1.shape[0]
        x = self.extract_feat(img)

        # For each level, we get the features for the two branches.
        extracted_features = x
        # extracted_features[0].detach_()
        split_extracted_features = [torch.split(x, n_batches, dim=0) for x in extracted_features]
        extracted_features_1 = [x[0] for x in split_extracted_features]
        extracted_features_2 = [x[1] for x in split_extracted_features]

        losses = dict()
        ##################################
        #        Tracking part           #
        ##################################
        siameserpn_losses, proposal_list_track = self.forward_track_train(
            n_batches,
            None,
            extracted_features_1,
            extracted_features_2,
            gt_bboxes1,
            gt_bboxes2,
            gt_trackids1,
            gt_trackids2,
            img_meta, )
        losses.update(siameserpn_losses)
        return losses


    def forward_train_graphNN(self,
                              img1,
                              img2,
                              img_meta,
                              gt_bboxes1,
                              gt_bboxes2,
                              gt_labels1,
                              gt_labels2,
                              gt_trackids1,
                              gt_trackids2,
                              gt_bboxes_ignore1=None,
                              gt_bboxes_ignore2=None,
                              gt_masks1=None,
                              gt_masks2=None,
                              proposals=None):
        ##################################
        #      Detection RPN part        #
        ##################################
        # same as two stage detector
        img = torch.cat([img1, img2], dim=0)
        n_batches = img1.shape[0]
        x = self.extract_feat(img)

        # For each level, we get the features for the two branches.
        extracted_features = x
        # extracted_features[0].detach_()
        split_extracted_features = [torch.split(x, n_batches, dim=0) for x in extracted_features]
        extracted_features_1 = [x[0] for x in split_extracted_features]
        extracted_features_2 = [x[1] for x in split_extracted_features]

        gt_bboxes_ignore = None
        gt_masks = None

        ##################################
        #         GraphNN part           #
        ##################################
        augmented_feats1 = [None for _ in range(len(extracted_features_1))]
        for idx, ext_feat1 in enumerate(extracted_features_1):
            space_time_mem = torch.stack((extracted_features_2[idx], ext_feat1), dim=0)
            augmented_feats1[idx] = self.space_time_modules[idx](space_time_mem, ext_feat1)

        # Set image train samples.
        gt_bboxes = gt_bboxes1
        gt_labels = gt_labels1
        img_train_meta = img_meta
        img_train_feat = augmented_feats1

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_list, losses = self.forward_rpn(img_train_feat, img_train_meta, gt_bboxes, gt_bboxes_ignore,
                                                     losses)
        else:
            proposal_list = proposals

        # TODO separate tracking and detection training.
        ##################################
        #        Tracking part           #
        ##################################
        '''
        siameserpn_losses, proposal_list_track = self.forward_track_train(
            n_batches,
            proposal_list,
            extracted_features_1,
            extracted_features_2,
            gt_bboxes1,
            gt_bboxes2,
            gt_trackids1,
            gt_trackids2,
            img_meta, )
        losses.update(siameserpn_losses)
        '''
        ##################################
        #           RCNN part            #
        ##################################
        if self.train_rcnn:
            num_imgs = n_batches
            losses = self.forward_rcnn_train(num_imgs,
                                             img_train_feat,
                                             proposal_list,
                                             gt_bboxes,
                                             gt_labels,
                                             gt_bboxes_ignore,
                                             gt_masks,
                                             losses)
        return losses

    def forward_train(self,
                        img1,
                        img2,
                        img_meta,
                        gt_bboxes1,
                        gt_bboxes2,
                        gt_labels1,
                        gt_labels2,
                        gt_trackids1,
                        gt_trackids2,
                        gt_bboxes_ignore1=None,
                        gt_bboxes_ignore2=None,
                        gt_masks1=None,
                        gt_masks2=None,
                        proposals=None):
        ##################################
        #      Detection RPN part        #
        ##################################
        # same as two stage detector
        img = torch.cat([img1, img2], dim=0)
        n_batches = img1.shape[0]
        x = self.extract_feat(img)

        # For each level, we get the features for the two branches.
        extracted_features = x
        # extracted_features[0].detach_()
        split_extracted_features = [torch.split(x, n_batches, dim=0) for x in extracted_features]
        extracted_features_1 = [x[0] for x in split_extracted_features]
        extracted_features_2 = [x[1] for x in split_extracted_features]

        gt_bboxes_ignore = None
        gt_masks = None

        # Set image train samples.
        gt_bboxes = gt_bboxes1
        gt_labels = gt_labels1
        img_train_meta = img_meta
        img_train_feat = extracted_features_1

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_list, losses = self.forward_rpn(img_train_feat, img_train_meta, gt_bboxes, gt_bboxes_ignore, losses)
        else:
            proposal_list = proposals

        ##################################
        #        Tracking part           #
        ##################################
        siameserpn_losses, proposal_list_track = self.forward_track_train(
            n_batches,
            proposal_list,
            extracted_features_1,
            extracted_features_2,
            gt_bboxes1,
            gt_bboxes2,
            gt_trackids1,
            gt_trackids2,
            img_meta,)
        losses.update(siameserpn_losses)

        ##################################
        #           RCNN part            #
        ##################################
        if self.train_rcnn:
            num_imgs = n_batches
            losses = self.forward_rcnn_train(num_imgs,
                                             img_train_feat,
                                             proposal_list,
                                             gt_bboxes,
                                             gt_labels,
                                             gt_bboxes_ignore,
                                             gt_masks,
                                             losses)
        return losses

    def update_sequence_list(self, new_tuple):
        '''

        :param new_tuple: (feature,det_bboxes,det_labels,rois_tracked)
        :return:
        '''
        if self.sequence_buffer is None:
            self.sequence_buffer = []
        if self.sequence_counter%self.sequence_gap==0:
            if len(self.sequence_buffer)==self.sequence_buffer_length:
                self.sequence_buffer.pop(-1)
            self.sequence_buffer = [new_tuple]+self.sequence_buffer
            self.sequence_counter = 0
        self.sequence_counter+=1
        return self.sequence_buffer

    def add_mapped_bboxes(self, bboxes_mapped):
        if self.sequence_mapped_bboxes is None:
            self.sequence_mapped_bboxes = []
        self.sequence_mapped_bboxes.append(bboxes_mapped)
        return self.sequence_mapped_bboxes

    def add_mapped_bboxes_result(self, bboxes_mapped_result):
        if self.sequence_mapped_bboxes_result is None:
            self.sequence_mapped_bboxes_result = []
        self.sequence_mapped_bboxes_result.append(bboxes_mapped_result)
        return self.sequence_mapped_bboxes_result

    def multi_track(self, current_feature, img_meta, cfg, max_gap = None):
        list_of_proposal_list_siamese = []
        for ind, tpl in enumerate(self.sequence_buffer):
            if max_gap is not None and (ind+1)*self.sequence_gap>max_gap:
                break
            feature, det_bboxes, det_labels, rois_tracked = tpl
            if len(rois_tracked)>0:
                list_of_proposal_list_siamese.append(self.simple_test_siamese_rpn(
                    feature, current_feature, rois_tracked, img_meta, cfg))
        proposal_list_siamese = \
            [torch.cat(tensor_tuples, dim=0) for tensor_tuples in zip(*list_of_proposal_list_siamese)]
        # NMS.
        proposal_list_siamese = [nms(proposals, cfg.nms_thr)[0] for proposals in proposal_list_siamese]
        return proposal_list_siamese

    def multi_track_with_non_nms_proposals(self, current_feature, img_meta, cfg, max_gap = None):
        list_of_proposal_list_siamese = []
        list_of_non_suppressed_proposal_list_siamese = []
        for ind, tpl in enumerate(self.sequence_buffer):
            if max_gap is not None and (ind+1)*self.sequence_gap>max_gap:
                break
            feature, det_bboxes, det_labels, rois_tracked = tpl
            if len(rois_tracked)>0:
                proposals_, non_suppressed_proposals_ = self.simple_test_siamese_rpn_with_non_suppressed_output(
                    feature, current_feature, rois_tracked, img_meta, cfg)
                list_of_proposal_list_siamese.append(proposals_)
                list_of_non_suppressed_proposal_list_siamese.append(non_suppressed_proposals_)
        proposal_list_siamese = \
            [torch.cat(tensor_tuples, dim=0) for tensor_tuples in zip(*list_of_proposal_list_siamese)]
        proposal_list_siamese_non_suppressed = \
            [torch.cat(tensor_tuples, dim=0) for tensor_tuples in zip(*list_of_non_suppressed_proposal_list_siamese)]
        # NMS.
        proposal_list_siamese = [nms(proposals, cfg.nms_thr)[0] for proposals in proposal_list_siamese]
        return proposal_list_siamese, proposal_list_siamese_non_suppressed

    def siamese_rpn(self, feat1, feat2, rpn_rois_1, img_meta):
        cls_score, bbox_pred, target_ranges, target_metas = self.siameserpn_head(feat1, feat2, rpn_rois_1, img_meta)
        proposal_inputs = (len(feat1[0]), rpn_rois_1, cls_score, bbox_pred, target_metas,)
        bboxes_list, scores_list = self.siameserpn_head.get_bboxes(*proposal_inputs)
        proposals = self.siameserpn_head.get_rois_from_boxes(len(feat1[0]),
                                                             bboxes_list,
                                                             scores_list,
                                                             score_threshold=0)
        return proposals

    def multi_rois_extractor(self, current_feature, img_meta, proposal, list_features):
        cls_score_list = []
        assert len(current_feature)==1
        roi_tracked = proposal.clone()
        roi_tracked[1:] = roi_tracked[:4]
        roi_tracked[0] = 0
        roi_tracked = roi_tracked.view(-1, 5)
        roi_feats_list = []
        for feature in list_features:
            proposal_list = self.siamese_rpn(current_feature, feature, roi_tracked, img_meta)
            rois = bbox2roi(proposal_list)
            roi_feats = self.bbox_roi_extractor(feature, rois)
            assert len(roi_feats)==len(feature)
            roi_feats_list.append(roi_feats)
        # TODO: maybe test max output instead.
        return roi_feats_list

    def multi_boxes_det(self, current_feature, img_meta, proposal, cfg_track, track_backwards = True):
        cls_score_list = []
        assert len(current_feature)==1
        roi_tracked = proposal.clone()
        roi_tracked[1:] = roi_tracked[:4]
        roi_tracked[0] = 0
        roi_tracked = roi_tracked.view(-1, 5)
        # start from the newest.
        for tpl in self.sequence_buffer:
            feature, det_bboxes, det_labels, rois_tracked = tpl
            proposal_list = self.simple_test_siamese_rpn(current_feature, feature, roi_tracked, img_meta, cfg_track)
            if len(proposal_list[0])>0:
                # extract probability.
                cls_score, bbox_pred = self.simple_test_bboxes_scores(feature, proposal_list)
                cls_score_list.append(cls_score[0])
                if track_backwards:
                    roi_tracked[1:] = proposal_list[0][:4]
                    current_feature = feature

        # TODO: maybe test max output instead.
        return cls_score_list

    def reset_tracking(self):
        self.rois_tracked = None
        self.extracted_feat1 = None
        self.sequence_counter = 0
        self.sequence_non_supressed_proposals = None

    def nms_to_box_result(self, bboxes, labels, cfg):
        det_bboxes, det_labels = multiclass_nms(bboxes, labels, cfg.score_thr, cfg.nms, cfg.max_per_img)
        bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        return bbox_results

    def proposals_to_box_result(self, proposal_list, cfg):
        bboxes = proposal_list[0][:, :4]
        labels = proposal_list[0][:, 4:]
        det_bboxes = torch.cat((bboxes,labels), dim=-1)
        det_labels = det_bboxes.new_zeros((len(det_bboxes),))
        bbox_result = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        return bbox_result

    def proposals_nms(self, proposal_list, cfg):
        bboxes = proposal_list[0][:, :4].repeat(1, self.bbox_head.num_classes)
        labels = proposal_list[0][:, 4:].repeat(1, self.bbox_head.num_classes)
        labels[:, 2:] = 0
        bboxes, labels = multiclass_nms(bboxes, labels, cfg.score_thr, cfg.nms, cfg.max_per_img)
        proposal_list[0] = bboxes
        return proposal_list

    def simple_test(self, img, img_meta, proposals=None, rescale=False, out = False):
        if self.img_train is True:
            return self.simple_test_img(img, img_meta, proposals, rescale)
        elif self.vid_train is True:
            return self.simple_test_vid(img, img_meta, None, rescale)
        elif self.graphnn_train:
            return self.simple_test_graphnn_img(img, img_meta, proposals, rescale)
        else:
            return self.simple_test_vid_track(img, img_meta, proposals, rescale, out)

    def simple_test_img(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        #Timer = clock()
        #Timer.tic()
        x = self.extract_feat(img)
        #Timer.toc('extract_feat')
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
        #Timer.toc('rpn')
        det_bboxes, det_labels = self.simple_test_bboxes(x, img_meta, proposal_list, None, rescale=False)
        proposal_list = [torch.cat([det_bboxes, det_labels], dim=-1)]

        det_bboxes, det_labels = self.simple_test_bboxes(x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        #Timer.toc('rcnn')
        bbox_results = bbox2result(det_bboxes, det_labels,self.bbox_head.num_classes)
        #Timer.toc('box2result')

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

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

    # new one
    def simple_test_vid_track_(self, img, img_meta, proposals=None, rescale=False, out = False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        det_bbox_result = None
        trk_bbox_result = None
        bbox_results = None
        det_bboxes, det_labels = None, None
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        proposal_list_siamese = None
        rois_tracked_mapped = None
        if len(x)>1:
            merged_x = self.merge_features(x)
        else:
            merged_x = x
        if self.sequence_buffer is not None and len(self.sequence_buffer) > 0 and self.sequence_buffer[-1][-1] is not None:
            proposal_list_siamese, proposal_list_siamese_non_nms = self.multi_track_with_non_nms_proposals(merged_x, img_meta, self.test_cfg.siameserpn, max_gap=self.multi_track_max_gap)
            rois_tracked_mapped = proposal_list_siamese_non_nms[-1][:,:4].clone()
        # save mapped boxes
        if rois_tracked_mapped is not None:
            mapped_bboxes = rois_tracked_mapped.repeat(1, self.bbox_head.num_classes)
            if rescale:
                mapped_bboxes /= img_meta[0]['scale_factor']
            mapped_bboxes, self.last_det_labels = multiclass(mapped_bboxes, self.last_det_labels)
            mapped_bbox_results = bbox2result(mapped_bboxes, self.last_det_labels, self.bbox_head.num_classes)
        else:
            mapped_bboxes = proposal_list[0].new_zeros((0, 5))
            mapped_labels = proposal_list[0].new_zeros((0,), dtype=torch.long)
            mapped_bbox_results = bbox2result(mapped_bboxes, mapped_labels, self.bbox_head.num_classes)
        self.add_mapped_bboxes_result(mapped_bbox_results)
        # merge proposals
        if proposal_list_siamese is not None and len(proposal_list_siamese) > 0 and len(proposal_list_siamese[0]) > 0:
            if not out:
                trk_bbox_result = self.proposals_to_box_result(proposal_list_siamese, self.test_cfg.rcnn)
            proposal_list_siamese[0][:, -1]=1
            proposal_list[0] = torch.cat([proposal_list[0], proposal_list_siamese[0]], dim=0)
            # ####
            # nms rpns for both det and track #
            # proposal_list[0], _ = nms(proposal_list[0], self.test_cfg.merged_rpn.nms_thr)
            # nms rpns for both det only #
            _, inds = nms(proposal_list[0], self.test_cfg.merged_rpn.nms_thr)
            # ####
            #inds = inds[inds<len(proposal_list[0])]
            proposal_list[0] = torch.cat([proposal_list[0][inds,:], proposal_list_siamese[0]], dim=0)
            # ####
        # get all results.
        if len(proposal_list[0])>0:
            rois = bbox2roi(proposal_list)
            cls_score, bbox_pred = self.simple_test_rois_scores(merged_x, rois)
            det_bboxes, det_labels = self.bbox_head.get_det_bboxes(rois,
                                                                   cls_score,
                                                                   bbox_pred,
                                                                   img_meta[0]['img_shape'],
                                                                   img_meta[0]['scale_factor'],
                                                                   rescale=rescale,
                                                                   cfg=None)
            self.last_det_labels = det_labels
            det_bboxes, det_labels = multiclass_nms(det_bboxes, det_labels,
                                                    self.test_cfg.rcnn.score_thr,
                                                    self.test_cfg.rcnn.nms,
                                                    self.test_cfg.rcnn.max_per_img)
            self.update_sequence_list((merged_x, None, None, rois))
        else:
            det_bboxes = det_bboxes.new_empty(0, 5)
            det_labels = det_labels.new_empty(0, 1)
            self.last_det_labels = None
            self.update_sequence_list((merged_x, None, None, None))

        bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)

        if not out:
            return det_bbox_result, trk_bbox_result, bbox_results
        else:
            return bbox_results

    # original
    def simple_test_vid_track(self, img, img_meta, proposals=None, rescale=False, out = False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        det_bbox_result = None
        trk_bbox_result = None
        bbox_results = None
        det_bboxes, det_labels = None, None
        #Timer = clock()
        #Timer.tic()
        x = self.extract_feat(img)
        proposal_list_raw = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
        det_bboxes, det_labels = self.simple_test_bboxes(x, img_meta, proposal_list_raw, None, rescale=False)
        # prune proposals.
        proposal_threshold = self.test_cfg.rcnn_propose.score_thr
        det_bboxes = det_bboxes[:, 4:].contiguous()
        det_labels = det_labels[:, 1:].contiguous()
        max_bboxes=[]
        max_labels=[]
        for _ in range(len(det_bboxes)):
            v, ind = torch.max(det_labels[_, :], -1)
            if v>proposal_threshold:
                max_bboxes.append(det_bboxes[_, ind*4:(ind+1)*4])
                max_labels.append(det_labels[_, ind:(ind+1)])
        if len(max_bboxes)>0:
            det_bboxes = torch.stack(max_bboxes, dim=0)
            det_labels = torch.stack(max_labels, dim=0)
            proposal_list = [torch.cat([det_bboxes, det_labels], dim=-1)]
            proposal_list[0], _ = nms(proposal_list[0], self.test_cfg.rcnn_propose.nms.iou_thr)
        else:
            proposal_list = [det_bboxes.new_empty(0, 5)]
        if not out:
            det_bbox_result = self.proposals_to_box_result(proposal_list, self.test_cfg.rcnn_propose)

        proposal_list_siamese = None
        rois_tracked_mapped = None
        if len(x)>1:
            merged_x = self.merge_features(x[1:-1])
        else:
            merged_x = x
        if self.sequence_buffer is not None and len(self.sequence_buffer) > 0 and self.sequence_buffer[-1][-1] is not None:
            proposal_list_siamese, proposal_list_siamese_non_nms = self.multi_track_with_non_nms_proposals(merged_x, img_meta, self.test_cfg.siameserpn, max_gap=self.multi_track_max_gap)
            rois_tracked_mapped = proposal_list_siamese_non_nms[-1][:,:4].clone()
        # save mapped boxes
        if rois_tracked_mapped is not None:
            mapped_bboxes = rois_tracked_mapped.repeat(1, self.bbox_head.num_classes)
            if rescale:
                mapped_bboxes /= img_meta[0]['scale_factor']
            mapped_bboxes, self.last_det_labels = multiclass(mapped_bboxes, self.last_det_labels)
            mapped_bbox_results = bbox2result(mapped_bboxes, self.last_det_labels, self.bbox_head.num_classes)
        else:
            mapped_bboxes = det_bboxes.new_zeros((0, 5))
            mapped_labels = det_bboxes.new_zeros((0,), dtype=torch.long)
            mapped_bbox_results = bbox2result(mapped_bboxes, mapped_labels, self.bbox_head.num_classes)
        self.add_mapped_bboxes_result(mapped_bbox_results)
        # merge proposals
        if proposal_list_siamese is not None and len(proposal_list_siamese) > 0 and len(proposal_list_siamese[0]) > 0:
            if not out:
                trk_bbox_result = self.proposals_to_box_result(proposal_list_siamese, self.test_cfg.rcnn)
            proposal_list_siamese[0][:, -1]=1
            len_proposal_det = len(proposal_list[0])
            proposal_list[0] = torch.cat([proposal_list[0], proposal_list_siamese[0]], dim=0)
            # ####
            # nms rpns for both det and track #
            # proposal_list[0], _ = nms(proposal_list[0], self.test_cfg.merged_rpn.nms_thr)
            # nms rpns for both det only #
            _, inds = nms(proposal_list[0], self.test_cfg.merged_rpn.nms_thr)
            # ####
            inds = inds[inds<len_proposal_det]
            proposal_list[0] = torch.cat([proposal_list[0][inds,:], proposal_list_siamese[0]], dim=0)
            # ####
        # get all results.
        if len(proposal_list[0])>0:
            rois = bbox2roi(proposal_list)
            cls_score, bbox_pred = self.simple_test_rois_scores(x, rois)
            det_bboxes, det_labels = self.bbox_head.get_det_bboxes(rois,
                                                                   cls_score,
                                                                   None,
                                                                   img_meta[0]['img_shape'],
                                                                   img_meta[0]['scale_factor'],
                                                                   rescale=rescale,
                                                                   cfg=None)
            det_bboxes = det_bboxes.repeat(1, det_labels.size()[-1])
            self.last_det_labels = det_labels
            det_bboxes, det_labels = multiclass(det_bboxes, det_labels)
            self.update_sequence_list((merged_x, None, None, rois))
        else:
            det_bboxes = det_bboxes.new_empty(0, 5)
            det_labels = det_labels.new_empty(0, 1)
            self.last_det_labels = None
            self.update_sequence_list((merged_x, None, None, None))
        #Timer.toc()
        bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)

        if not out:
            return det_bbox_result, trk_bbox_result, bbox_results
        else:
            return bbox_results

    def retrieve_space_time_mem(self, idx=None):
        '''

        :param idx: None if all space time memories are used. Or list of indices to select.
        :return: list of memory for relation distillation.
        '''
        if idx is None:
            mems_list = self.space_time_mem
        else:
            mems_list = self.space_time_mem[idx]
        return mems_list

    def simple_test_graphnn_img(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        #Timer = clock()
        #Timer.tic()
        extracted_features_1 = self.extract_feat(img)
        # Augment
        augmented_feats1 = [None for _ in range(len(extracted_features_1))]
        space_time_mem_list = self.retrieve_space_time_mem(idx=None)
        #print('space_time_mem_list len:', len(space_time_mem_list))
        for idx, ext_feat1 in enumerate(extracted_features_1):
            space_time_mem = torch.stack([it[idx] for it in space_time_mem_list]+[ext_feat1], dim=0)
            augmented_feats1[idx] = self.space_time_modules[idx](space_time_mem, ext_feat1)

        # update memory.
        if self.space_time_mem_counter%5==0:
            if len(self.space_time_mem)==20:
                self.space_time_mem = self.space_time_mem[:-1]
            self.space_time_mem = self.space_time_mem + [augmented_feats1]
            self.space_time_mem_counter = 0
        self.space_time_mem_counter+=1
        #Timer.toc('extract_feat')
        x = augmented_feats1
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
        #Timer.toc('rpn')
        det_bboxes, det_labels = self.simple_test_bboxes(x, img_meta, proposal_list, None, rescale=False)
        proposal_list = [torch.cat([det_bboxes, det_labels], dim=-1)]

        det_bboxes, det_labels = self.simple_test_bboxes(x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        #Timer.toc('rcnn')
        bbox_results = bbox2result(det_bboxes, det_labels,self.bbox_head.num_classes)
        #Timer.toc('box2result')

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    # TODO need to modify.
    def simple_test_graphnn_track(self, img, img_meta, proposals=None, rescale=False, out = False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        det_bbox_result = None
        trk_bbox_result = None
        bbox_results = None
        det_bboxes, det_labels = None, None
        # Timer = clock()
        # Timer.tic()
        x = self.extract_feat(img)
        proposal_list_raw = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
        det_bboxes, det_labels = self.simple_test_bboxes(x, img_meta, proposal_list_raw, None, rescale=False)
        # prune proposals.
        proposal_threshold = self.test_cfg.rcnn_propose.score_thr
        det_bboxes = det_bboxes[:, 4:].contiguous()
        det_labels = det_labels[:, 1:].contiguous()
        max_bboxes = []
        max_labels = []
        for _ in range(len(det_bboxes)):
            v, ind = torch.max(det_labels[_, :], -1)
            if v > proposal_threshold:
                max_bboxes.append(det_bboxes[_, ind * 4:(ind + 1) * 4])
                max_labels.append(det_labels[_, ind:(ind + 1)])
        if len(max_bboxes) > 0:
            det_bboxes = torch.stack(max_bboxes, dim=0)
            det_labels = torch.stack(max_labels, dim=0)
            proposal_list = [torch.cat([det_bboxes, det_labels], dim=-1)]
            proposal_list[0], _ = nms(proposal_list[0], self.test_cfg.rcnn_propose.nms.iou_thr)
        else:
            proposal_list = [det_bboxes.new_empty(0, 5)]
        if not out:
            det_bbox_result = self.proposals_to_box_result(proposal_list, self.test_cfg.rcnn_propose)

        proposal_list_siamese = None
        rois_tracked_mapped = None
        if len(x) > 1:
            merged_x = self.merge_features(x[1:-1])
        else:
            merged_x = x
        if self.sequence_buffer is not None and len(self.sequence_buffer) > 0 and self.sequence_buffer[-1][
            -1] is not None:
            proposal_list_siamese, proposal_list_siamese_non_nms = self.multi_track_with_non_nms_proposals(merged_x,
                                                                                                           img_meta,
                                                                                                           self.test_cfg.siameserpn,
                                                                                                           max_gap=self.multi_track_max_gap)
            rois_tracked_mapped = proposal_list_siamese_non_nms[-1][:, :4].clone()
        # save mapped boxes
        if rois_tracked_mapped is not None:
            mapped_bboxes = rois_tracked_mapped.repeat(1, self.bbox_head.num_classes)
            if rescale:
                mapped_bboxes /= img_meta[0]['scale_factor']
            mapped_bboxes, self.last_det_labels = multiclass(mapped_bboxes, self.last_det_labels)
            mapped_bbox_results = bbox2result(mapped_bboxes, self.last_det_labels, self.bbox_head.num_classes)
        else:
            mapped_bboxes = det_bboxes.new_zeros((0, 5))
            mapped_labels = det_bboxes.new_zeros((0,), dtype=torch.long)
            mapped_bbox_results = bbox2result(mapped_bboxes, mapped_labels, self.bbox_head.num_classes)
        self.add_mapped_bboxes_result(mapped_bbox_results)
        # merge proposals
        if proposal_list_siamese is not None and len(proposal_list_siamese) > 0 and len(proposal_list_siamese[0]) > 0:
            if not out:
                trk_bbox_result = self.proposals_to_box_result(proposal_list_siamese, self.test_cfg.rcnn)
            proposal_list_siamese[0][:, -1] = 1
            len_proposal_det = len(proposal_list[0])
            proposal_list[0] = torch.cat([proposal_list[0], proposal_list_siamese[0]], dim=0)
            # ####
            # nms rpns for both det and track #
            # proposal_list[0], _ = nms(proposal_list[0], self.test_cfg.merged_rpn.nms_thr)
            # nms rpns for both det only #
            _, inds = nms(proposal_list[0], self.test_cfg.merged_rpn.nms_thr)
            # ####
            inds = inds[inds < len_proposal_det]
            proposal_list[0] = torch.cat([proposal_list[0][inds, :], proposal_list_siamese[0]], dim=0)
            # ####
        # get all results.
        if len(proposal_list[0]) > 0:
            rois = bbox2roi(proposal_list)
            cls_score, bbox_pred = self.simple_test_rois_scores(x, rois)
            det_bboxes, det_labels = self.bbox_head.get_det_bboxes(rois,
                                                                   cls_score,
                                                                   None,
                                                                   img_meta[0]['img_shape'],
                                                                   img_meta[0]['scale_factor'],
                                                                   rescale=rescale,
                                                                   cfg=None)
            det_bboxes = det_bboxes.repeat(1, det_labels.size()[-1])
            self.last_det_labels = det_labels
            det_bboxes, det_labels = multiclass(det_bboxes, det_labels)
            self.update_sequence_list((merged_x, None, None, rois))
        else:
            det_bboxes = det_bboxes.new_empty(0, 5)
            det_labels = det_labels.new_empty(0, 1)
            self.last_det_labels = None
            self.update_sequence_list((merged_x, None, None, None))
        # Timer.toc()
        bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)

        if not out:
            return det_bbox_result, trk_bbox_result, bbox_results
        else:
            return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = list(self.extract_feats(imgs))
        proposal_list = self.aug_test_rpn(
            x, img_metas, self.test_cfg.rpn)

        det_bboxes, det_labels = self.aug_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg.rcnn)

        proposal_list_siamese = None
        if self.rois_tracked is not None and len(self.rois_tracked) > 0 and self.extracted_feat1 is not None:
            bboxes = self.rois_tracked[:, 1:].cpu().numpy()
            # prepare bboxes list to be tracked.
            rpn_rois_list = []
            for i in range(len(x)):
                scale_factor = img_metas[i][0]['scale_factor']
                flip = img_metas[i][0]['flip']
                img_shape = img_metas[i][0]['img_shape']
                rpn_rois = np.zeros((len(bboxes), 5), np.float32)
                transformed_boxes = self.bbox_transform(bboxes, img_shape, scale_factor, flip)
                for idx, box in enumerate(transformed_boxes):
                    rpn_rois[idx, 1:] = box[:]
                rpn_rois_list.append(self.rois_tracked.new_tensor(rpn_rois))
            proposal_list_siamese = self.aug_test_siamese_rpn(self.extracted_feat1, x, rpn_rois_list, img_metas, self.test_cfg.siameserpn)

        if proposal_list_siamese is not None and len(proposal_list_siamese[0])>0:
            det_bboxes_track, det_labels_track = self.aug_test_bboxes(x, img_metas, proposal_list_siamese, self.test_cfg.rcnn)
            if len(proposal_list_siamese)>0:
                print('track:%d'%(len(proposal_list_siamese[0])))
                det_bboxes = proposal_list_siamese[0] #det_bboxes_track
                det_labels = det_labels.new_zeros((len(proposal_list_siamese[0]))) #det_labels_track
            else:
                print('lost')
                self.reset_tracking()
                return None

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        self.extracted_feat1 = x
        self.rois_tracked = det_bboxes.new_zeros((len(det_bboxes), 5))
        for idx, box in enumerate(det_bboxes):
            self.rois_tracked[idx, 1:] = box[:4]

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results

