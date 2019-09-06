from ..registry import DETECTORS
from .two_stage import TwoStageDetector
from .. import builder

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, auto_fp16, bbox2delta, delta2bbox
from .test_mixins import SiameseRPNTestMixin

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
    def __init__(self, siameserpn_head,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
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
        #self.nms = trNMS(cfg.SIAMESE.PANELTY_K, cfg.SIAMESE.HANNING_WINDOW_WEIGHT, cfg.SIAMESE.HANNING_WINDOW_SIZE_FACTOR)

    '''@auto_fp16(apply_to=('img',))
    def forward(self, img1, img_meta, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)'''

    @auto_fp16(apply_to=('img',))
    def forward(self, return_loss = True, **inputs):
        if return_loss is True:
            return self.forward_train(**inputs)
        else:
            return self.forward_test(inputs.pop('img', None), inputs.pop('img_meta', None), **inputs)

    def _get_mapper_gt1_to_gt2(self, gt_bboxes_1, gt_ids_1, gt_bboxes_2, gt_ids_2):
        def mapper(img_id, id1_query):
            box1s = gt_bboxes_1[img_id]
            id1s = gt_ids_1[img_id]
            box2s = gt_bboxes_2[img_id]
            id2s =  gt_ids_2[img_id]
            mapped_gt2_box = None
            mapped_gt2_id = None
            for idx_1 in range(len(id1s)):
                id1 = id1s[idx_1]
                if id1==id1_query:
                    for idx_2 in range(len(id2s)):
                        id2 = id2s[idx_2]
                        if id2==id1:
                            mapped_gt2_box = box2s[idx_2:idx_2+1,:]
                            mapped_gt2_id = gt_ids_2[idx_2:idx_2+1]
                            return mapped_gt2_box, mapped_gt2_id
            return mapped_gt2_box, mapped_gt2_id
        return mapper

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
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
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
        #        Detection part          #
        ##################################
        # same as two stage detector
        img = torch.cat([img1, img2], dim=0)
        gt_bboxes = gt_bboxes1 + gt_bboxes2
        gt_labels = gt_labels1 + gt_labels2
        gt_bboxes_ignore = None
        gt_masks = None
        n_batches = img1.shape[0]

        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_list, losses = self.forward_rpn(x, img_meta*2, gt_bboxes, gt_bboxes_ignore, losses)
        else:
            proposal_list = proposals

        num_imgs = n_batches*2
        losses = self.forward_rcnn_train(num_imgs,
                     x,
                     proposal_list,
                     gt_bboxes,
                     gt_labels,
                     gt_bboxes_ignore,
                     gt_masks,
                     losses)

        ##################################
        #        Tracking part           #
        ##################################
        # For each level, we get the features for the two branches.
        extracted_features = x
        split_extracted_features = [torch.split(x, n_batches, dim=0) for x in extracted_features]
        extracted_features_1 = tuple([x[0] for x in split_extracted_features])
        extracted_features_2 = tuple([x[1] for x in split_extracted_features])
        # Get rpn proposals.
        proposal_list_1 = proposal_list[:n_batches]
        proposal_list_2 = proposal_list[n_batches:]
        # Get training rois.
        tracking_bbox_assigner = build_assigner(self.train_cfg.siameserpn.assigner_track)
        tracking_bbox_sampler = build_sampler(self.train_cfg.siameserpn.sampler_track, context=self)
        sampling_results = []
        for i in range(n_batches):
            assign_result_1 = tracking_bbox_assigner.assign(proposal_list_1[i],
                                                            gt_bboxes1[i],
                                                            None,
                                                            gt_trackids1[i])
            sampling_result_1 = tracking_bbox_sampler.sample(assign_result_1,
                                                             proposal_list_1[i],
                                                             gt_bboxes1[i],
                                                             gt_trackids1[i])

            sampling_results.append(sampling_result_1)

        # Only pos boxes are used.
        rpn_rois_1 = bbox2roi([res.pos_bboxes for res in sampling_results])
        mapper_gt1_to_gt2 = self._get_mapper_gt1_to_gt2(gt_bboxes1, gt_trackids1, gt_bboxes2, gt_trackids2)
        siameserpn_gt_boxes = []
        for i in range(n_batches):
            res = sampling_results[i]
            lbls = res.pos_gt_labels
            boxes = res.pos_bboxes
            gt_boxes = res.pos_gt_bboxes
            for idx in range(len(lbls)):
                lbl = lbls[idx]
                box = boxes[idx:idx+1]
                gt_box = gt_boxes[idx:idx+1]
                mapped_gt2_box, mapped_gt2_id = mapper_gt1_to_gt2(i, lbl)
                gt_shifted = delta2bbox(mapped_gt2_box, bbox2delta(box, gt_box)) \
                    if mapped_gt2_box is not None else mapped_gt2_box
                siameserpn_gt_boxes.append(gt_shifted)

        siameserpn_outs = self.siameserpn_head(extracted_features_1, extracted_features_2, rpn_rois_1, img_meta)
        siameserpn_loss_inputs = siameserpn_outs + (siameserpn_gt_boxes, self.train_cfg.siameserpn)
        siameserpn_losses = self.siameserpn_head.loss(*siameserpn_loss_inputs, gt_bboxes_ignore=None)
        losses.update(siameserpn_losses)
        return losses

    def reset_tracking(self):
        self.rois_tracked = None
        self.extracted_feat1 = None

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        proposal_list_siamese = None
        if self.rois_tracked is not None and self.extracted_feat1 is not None:
            print('len(self.rois_tracked):', len(self.rois_tracked))
            proposal_list_siamese = self.simple_test_siamese_rpn(
                x, self.extracted_feat1, self.rois_tracked, img_meta, self.test_cfg.siameserpn)
            print('proposal_list_siamese:',proposal_list_siamese)
            '''
            for img_id, bboxes in enumerate(proposal_list):
                # add siamese proposal_list to proposal_list if it is not empty.
                siamese_bboxes = proposal_list_siamese[img_id]
                if bboxes.size(0) > 0 and siamese_bboxes.size(0) > 0:
                    proposal_list[img_id] = torch.cat([bboxes[:, :4], siamese_bboxes[:, :4]], dim=-1)
                elif siamese_bboxes.size(0) > 0:
                    proposal_list[img_id] = siamese_bboxes
                else:
                    continue'''
            #print(len(proposal_list), proposal_list[0].shape)
            #print(len(proposal_list_siamese), proposal_list_siamese[0].shape)

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)

        if proposal_list_siamese is not None:
            det_bboxes_track, det_labels_track = self.simple_test_bboxes(
                x, img_meta, proposal_list_siamese, self.test_cfg.rcnn, rescale=rescale)
            if len(det_bboxes_track)>0:
                print('track')
                #det_bboxes, det_labels = det_bboxes_track, det_labels_track
                #print('det_bboxes_track:', det_bboxes_track)
                #print('proposal_list_siamese[0]:', proposal_list_siamese[0])
                det_bboxes, det_labels = proposal_list_siamese[0], det_labels_track.new_zeros((len(proposal_list_siamese[0])))
            else:
                print('lost')
                self.reset_tracking()
                return None


        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        print('det_bboxes:',det_bboxes)
        #print(len(det_bboxes))
        self.extracted_feat1 = x
        self.rois_tracked = det_bboxes.new_zeros((len(det_bboxes), 5))
        for idx, box in enumerate(det_bboxes):
            self.rois_tracked[idx, 1:] = box[:4]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results

