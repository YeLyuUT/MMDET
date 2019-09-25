import logging
from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch.nn as nn
import cv2
from matplotlib import pyplot as plt

from mmdet.core import auto_fp16, get_classes, tensor2imgs


class BaseDetector(nn.Module):
    """Base class for detectors"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, imgs, img_metas, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_meta, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)

    def vis_detections(self, ax, im, bboxes, labels, class_names, thresh, clr='g'):
        im = im[:, :, (2, 1, 0)]
        ax.imshow(im, aspect='auto')
        if bboxes is None:
            return True
        for i in range(bboxes.shape[0]):
            score = bboxes[i, -1]
            bbox = bboxes[i, :]
            class_name = class_names[labels[i]]
            if score > thresh:
                ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor=clr, linewidth=1)
                )
                ax.text(
                    bbox[0], bbox[1]+11,
                    class_name+' %.2f'%(bbox[-1]),
                    #'object %.2f' % (bbox[-1]),
                    fontsize=10,
                    family='serif',
                    bbox=dict(
                        facecolor=clr,  # if classes[i]==2 else 'r',
                        alpha=0.4, pad=0, edgecolor='none'),
                    color='white')

    def show_bbox_result_custom(self, fig, ax, bbox_result, img_show, class_names, score_thr, winname='', waittime=0,clr='g'):
        if bbox_result is not None:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
        else:
            bboxes, labels = None, None
        #print('input image.shape:', img_show.shape)
        self.vis_detections(ax, img_show, bboxes, labels, class_names, score_thr, clr=clr)


    def show_bbox_result_default(self, bbox_result, img_show, class_names, score_thr):
        bboxes = np.vstack(bbox_result)
        # draw bounding boxes
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        mmcv.imshow_det_bboxes(img_show,
                               bboxes,
                               labels,
                               class_names=class_names,
                               score_thr=score_thr)

    def show_result(self,
                    data,
                    result,
                    img_norm_cfg,
                    dataset=None,
                    score_thr=0.1,
                    use_custom_vis = True):
        det_bbox_result, trk_bbox_result = None, None
        if isinstance(result, tuple):
            if len(result)==2:
                bbox_result, segm_result = result
            elif len(result)==3:
                det_bbox_result, trk_bbox_result, bbox_result = result
                segm_result = None
            elif len(result)==4:
                det_bbox_result, trk_bbox_result, bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None

        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_norm_cfg)
        assert len(imgs) == len(img_metas)

        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))
        if use_custom_vis:
            dpi = 100.0
            fig,(ax_det,ax_trk,ax) = plt.subplots(1,3, frameon=False, dpi=dpi)
            fig.set_size_inches(imgs[0].shape[1]*3 / dpi, imgs[0].shape[0] / dpi)
            ax.axis('off')
            #if det_bbox_result is not None:
            ax_det.axis('off')
            #if trk_bbox_result is not None:
            ax_trk.axis('off')
        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]
            if bbox_result is None:
                cv2.imshow('', img_show)
                cv2.waitKey(0)
                continue
            if use_custom_vis:
                #if det_bbox_result is not None:
                    #print('det_bbox_result:', det_bbox_result)
                self.show_bbox_result_custom(fig, ax_det, det_bbox_result, img_show, class_names, score_thr, winname='det_bbox_result', waittime=1,clr='y')
                #if trk_bbox_result is not None:
                    #print('trk_bbox_result:', trk_bbox_result)
                self.show_bbox_result_custom(fig, ax_trk, trk_bbox_result, img_show, class_names, score_thr, winname='trk_bbox_result', waittime=1,clr='m')
                #print('bbox_result:', bbox_result)
                self.show_bbox_result_custom(fig, ax, bbox_result, img_show, class_names, score_thr, winname='bbox_result',clr='g')
                plt.tight_layout(0, h_pad=0.1, w_pad=0.1)
                # convert canvas to image
                fig.canvas.draw()
                img_show = np.array(fig.canvas.renderer.buffer_rgba())
                img_show = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)
                plt.close(fig)
                #print('plot image.shape:', img_show.shape)
                cv2.imshow('', img_show)
                cv2.waitKey(0)
                #plt.show()
            else:
                self.show_bbox_result_default(bbox_result, img_show, class_names, score_thr)
