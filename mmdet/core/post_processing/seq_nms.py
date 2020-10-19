# --------------------------------------------------------
# Flow-Guided Feature Aggregation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified byYuqing Zhu, Xizhou Zhu
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import numpy as np

import profile
import cv2
import time
import os
from copy import deepcopy
import numpy as np
from functools import partial

CLASSES = ('airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus',
           'car', 'cattle', 'dog', 'domestic cat', 'elephant', 'fox',
           'giant panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey',
           'motorcycle', 'rabbit', 'red panda', 'sheep', 'snake', 'squirrel',
           'tiger', 'train', 'turtle', 'watercraft', 'whale', 'zebra')


NMS_THRESH = 0.45
MAX_THRESH = 1e-2
MIN_LENGTH = 2
RESCORE_PERCENTAGE = 0.5
RESCORE_TOP_N = 20

def createLinksWithMapper(dets_all, mapped_dets_all, IOU_THRESH = 0.5):
    links_all = []

    frame_num = len(dets_all[0])
    cls_num = len(CLASSES)

    for cls_ind in range(cls_num):
        links_cls = []
        for frame_ind in range(frame_num - 1):
            dets1 = mapped_dets_all[cls_ind][frame_ind+1]
            #_dets1 = dets_all[cls_ind][frame_ind]
            dets2 = dets_all[cls_ind][frame_ind + 1]
            box1_num = len(dets1)
            box2_num = len(dets2)

            areas1 = np.empty(box1_num)
            for box1_ind, box1 in enumerate(dets1):
                areas1[box1_ind] = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)

            areas2 = np.empty(box2_num)
            for box2_ind, box2 in enumerate(dets2):
                areas2[box2_ind] = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

            links_frame = []
            for box1_ind, box1 in enumerate(dets1):
                area1 = areas1[box1_ind]
                x1 = np.maximum(box1[0], dets2[:, 0])
                y1 = np.maximum(box1[1], dets2[:, 1])
                x2 = np.minimum(box1[2], dets2[:, 2])
                y2 = np.minimum(box1[3], dets2[:, 3])
                w = np.maximum(0.0, x2 - x1 + 1)
                h = np.maximum(0.0, y2 - y1 + 1)
                inter = w * h
                ovrs = inter / (area1 + areas2 - inter)
                links_box = [ovr_ind for ovr_ind, ovr in enumerate(ovrs) if
                             ovr >= IOU_THRESH]
                links_frame.append(links_box)
            links_cls.append(links_frame)
        links_all.append(links_cls)
    return links_all

def createLinks(dets_all, IOU_THRESH = 0.5):
    links_all = []

    frame_num = len(dets_all[0])
    cls_num = len(CLASSES)
    for cls_ind in range(cls_num):
        links_cls = []
        for frame_ind in range(frame_num - 1):
            dets1 = dets_all[cls_ind][frame_ind]
            dets2 = dets_all[cls_ind][frame_ind + 1]
            box1_num = len(dets1)
            box2_num = len(dets2)
            if frame_ind == 0:
                areas1 = np.empty(box1_num)
                for box1_ind, box1 in enumerate(dets1):
                    areas1[box1_ind] = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
            else:
                areas1 = areas2

            areas2 = np.empty(box2_num)
            for box2_ind, box2 in enumerate(dets2):
                areas2[box2_ind] = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

            links_frame = []
            for box1_ind, box1 in enumerate(dets1):
                area1 = areas1[box1_ind]
                x1 = np.maximum(box1[0], dets2[:, 0])
                y1 = np.maximum(box1[1], dets2[:, 1])
                x2 = np.minimum(box1[2], dets2[:, 2])
                y2 = np.minimum(box1[3], dets2[:, 3])
                w = np.maximum(0.0, x2 - x1 + 1)
                h = np.maximum(0.0, y2 - y1 + 1)
                inter = w * h
                ovrs = inter / (area1 + areas2 - inter)
                links_box = [ovr_ind for ovr_ind, ovr in enumerate(ovrs) if
                             ovr >= IOU_THRESH]
                links_frame.append(links_box)
            links_cls.append(links_frame)
        links_all.append(links_cls)
    return links_all

# standard
def rescore(dets, rootindex, maxpath, maxsum):
    newscore = maxsum / len(maxpath)
    for i, box_ind in enumerate(maxpath):
        dets[rootindex + i][box_ind][4] = newscore

def maxPath(dets_all, links_all, rescoreFunc = rescore):
    for cls_ind, links_cls in enumerate(links_all):

        delete_sets=[[]for i in range(0,len(dets_all[0]))]
        delete_single_box=[]
        dets_cls = dets_all[cls_ind]

        num_path=0
        # compute the number of links
        sum_links=0
        for frame_ind, frame in enumerate(links_cls):
            for box_ind,box in enumerate(frame):
                sum_links+=len(box)

        while True:
            num_path+=1

            rootindex, maxpath, maxsum = findMaxPath(links_cls, dets_cls, delete_single_box)

            if (maxsum<MAX_THRESH or sum_links==0 or len(maxpath) <1):
                break
            if (len(maxpath)==1):
                delete=[rootindex,maxpath[0]]
                delete_single_box.append(delete)
            rescoreFunc(dets_cls, rootindex, maxpath, maxsum)
            t4=time.time()
            delete_set,num_delete=deleteLink(dets_cls, links_cls, rootindex, maxpath, NMS_THRESH)
            sum_links-=num_delete
            for i, box_ind in enumerate(maxpath):
                if box_ind not in delete_set[i]:
                    print('warning,', box_ind, 'not in delete_set',i, delete_set[i])
                else:
                    pass
                delete_set[i].remove(box_ind)
                delete_single_box.append([[rootindex+i],box_ind])
                for j in delete_set[i]:
                    dets_cls[i+rootindex][j]=np.zeros(5)
                delete_sets[i+rootindex]=delete_sets[i+rootindex]+delete_set[i]

        for frame_idx,frame in enumerate(dets_all[cls_ind]):            
            a=range(0,len(frame))
            keep=list(set(a).difference(set(delete_sets[frame_idx])))
            dets_all[cls_ind][frame_idx]=frame[keep,:]


    return dets_all


def findMaxPath(links,dets,delete_single_box):

    len_dets=[len(dets[i]) for i in range(len(dets))]
    max_boxes=np.max(len_dets)
    num_frame=len(links)+1
    a=np.zeros([num_frame,max_boxes])
    new_dets=np.zeros([num_frame,max_boxes])
    for delete_box in delete_single_box:
        new_dets[delete_box[0],delete_box[1]]=1
    if(max_boxes==0):
        max_path=[]
        return 0,max_path,0

    b=np.full((num_frame,max_boxes),-1)
    for l in range(len(dets)):
        for j in range(len(dets[l])):
            if(new_dets[l,j]==0):
                a[l,j]=dets[l][j][-1]



    for i in range(1,num_frame):
        l1=i-1;
        for box_id,box in enumerate(links[l1]):
            for next_box_id in box:

                weight_new=a[i-1,box_id]+dets[i][next_box_id][-1]
                if(weight_new>a[i,next_box_id]):
                    a[i,next_box_id]=weight_new
                    b[i,next_box_id]=box_id

    i,j=np.unravel_index(a.argmax(),a.shape)

    maxpath=[j]
    maxscore=a[i,j]
    while(b[i,j]!=-1):

            maxpath.append(b[i,j])
            j=b[i,j]
            i=i-1


    rootindex=i
    maxpath.reverse()
    return rootindex, maxpath, maxscore

# large percentage
def rescore_percentage(dets, rootindex, maxpath, maxsum, percentage=RESCORE_PERCENTAGE, use_add=False):
    scores = []
    for i, box_ind in enumerate(maxpath):
        scores.append(dets[rootindex + i][box_ind][4])
    scores = sorted(scores)
    if percentage<0.9999:
        if len(scores)>=MIN_LENGTH:
            length = min(int(len(scores)*percentage)+1, len(scores))
            newscore = sum(scores[-length:])/length
        else:
            newscore = sum(scores)/MIN_LENGTH
    else:
        newscore = maxsum / len(maxpath)
    for i, box_ind in enumerate(maxpath):
        if use_add:
            dets[rootindex + i][box_ind][4] = newscore + dets[rootindex + i][box_ind][4]
        else:
            dets[rootindex + i][box_ind][4] = newscore

# top N.
def rescore_top_n(dets, rootindex, maxpath, maxsum, top_n=RESCORE_TOP_N, use_add=False):
    scores = []
    for i, box_ind in enumerate(maxpath):
        scores.append(dets[rootindex + i][box_ind][4])
    scores = sorted(scores)
    if len(scores)>=MIN_LENGTH:
        length = min(top_n, len(scores))
        newscore = sum(scores[-length:])/length
    else:
        newscore = sum(scores) / MIN_LENGTH
    for i, box_ind in enumerate(maxpath):
        if use_add:
            dets[rootindex + i][box_ind][4] = newscore + dets[rootindex + i][box_ind][4]
        else:
            dets[rootindex + i][box_ind][4] = newscore

def deleteLink(dets, links, rootindex, maxpath, thesh):

    delete_set=[]
    num_delete_links=0

    for i, box_ind in enumerate(maxpath):
        areas = [(box[2] - box[0] + 1) * (box[3] - box[1] + 1) for box in dets[rootindex + i]]
        area1 = areas[box_ind]
        box1 = dets[rootindex + i][box_ind]
        x1 = np.maximum(box1[0], dets[rootindex + i][:, 0])
        y1 = np.maximum(box1[1], dets[rootindex + i][:, 1])
        x2 = np.minimum(box1[2], dets[rootindex + i][:, 2])
        y2 = np.minimum(box1[3], dets[rootindex + i][:, 3])
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        inter = w * h

        ovrs = inter / (area1 + areas - inter)
        #saving the box need to delete
        deletes = [ovr_ind for ovr_ind, ovr in enumerate(ovrs) if ovr >= thesh]
        delete_set.append(deletes)

        #delete the links except for the last frame
        if rootindex + i < len(links):
            for delete_ind in deletes:
                num_delete_links+=len(links[rootindex+i][delete_ind])
                links[rootindex + i][delete_ind] = []

        if i > 0 or rootindex > 0:

            #delete the links which point to box_ind
            for priorbox in links[rootindex + i - 1]:
                for delete_ind in deletes:
                    if delete_ind in priorbox:
                        priorbox.remove(delete_ind)
                        num_delete_links+=1

    return delete_set,num_delete_links

def seq_nms(dets, IOU_THRESH = 0.5):
    links = createLinks(dets, IOU_THRESH)
    dets = maxPath(dets, links)
    return dets
    
def seq_nms_05(dets, IOU_THRESH = 0.5):
    links = createLinks(dets, IOU_THRESH)
    dets = maxPath(dets, links, partial(rescore_percentage, percentage=0.5))
    return dets

def seq_nms_025(dets, IOU_THRESH = 0.5):
    links = createLinks(dets, IOU_THRESH)
    dets = maxPath(dets, links, partial(rescore_percentage, percentage=0.25))
    return dets

def seq_nms_with_mapper(dets, mapped_dets, IOU_THRESH = 0.5):
    links = createLinksWithMapper(dets, mapped_dets, IOU_THRESH)
    dets = maxPath(dets, links)
    return dets
    
def seq_nms_with_mapper_05(dets, mapped_dets, IOU_THRESH = 0.5):
    links = createLinksWithMapper(dets, mapped_dets, IOU_THRESH)
    dets = maxPath(dets, links, partial(rescore_percentage, percentage=0.5))
    return dets

def seq_nms_with_mapper_multiple(dets, mapped_dets):
    links = createLinksWithMapper(dets, mapped_dets)

    dets_0 = deepcopy(dets)
    links_0 = deepcopy(links)
    dets_0 = maxPath(dets_0, links_0, rescore)

    dets_1 = deepcopy(dets)
    links_1 = deepcopy(links)
    dets_1 = maxPath(dets_1, links_1, partial(rescore_percentage, percentage=0.25))

    dets_2 = deepcopy(dets)
    links_2 = deepcopy(links)
    dets_2 = maxPath(dets_2, links_2, partial(rescore_percentage, percentage=0.5))

    dets_3 = deepcopy(dets)
    links_3 = deepcopy(links)
    dets_3 = maxPath(dets_3, links_3, partial(rescore_top_n, top_n=1))

    dets_4 = deepcopy(dets)
    links_4 = deepcopy(links)
    dets_4 = maxPath(dets_4, links_4, partial(rescore_top_n, top_n=10))

    dets_5 = deepcopy(dets)
    links_5 = deepcopy(links)
    dets_5 = maxPath(dets_5, links_5, partial(rescore_top_n, top_n=20))

    dets_6 = deepcopy(dets)
    links_6 = deepcopy(links)
    dets_6 = maxPath(dets_6, links_6, partial(rescore_percentage, percentage=0.25, use_add=True))

    dets_7 = deepcopy(dets)
    links_7 = deepcopy(links)
    dets_7 = maxPath(dets_7, links_7, partial(rescore_percentage, percentage=0.5, use_add=True))

    dets_8 = deepcopy(dets)
    links_8 = deepcopy(links)
    dets_8 = maxPath(dets_8, links_8, partial(rescore_percentage, percentage=1.0, use_add=True))

    return dets_0, dets_1, dets_2, dets_3, dets_4, dets_5, dets_6, dets_7, dets_8