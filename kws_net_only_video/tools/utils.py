#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import logging
import numpy as np
import torch
import os
import copy

def get_logger(filename):
    # Logging configuration: set the basic configuration of the logging system
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s',
                                      datefmt='%m-%d %H:%M')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # File logger
    file_handler = logging.FileHandler("{}.log".format(filename))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    # Stderr logger
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    std_handler.setLevel(logging.DEBUG)
    logger.addHandler(std_handler)
    return logger


def cal_indicator(pre,label):
    TP = 0.0
    FN = 0.0
    TN = 0.0
    FP = 0.0
    for i, it in enumerate(pre):
        if it == 1.0 and label[i] == 1.0:
            TP += 1.0
        elif it == 1.0 and label[i] == -0.0:
            FP += 1
        elif it == -0.0 and label[i] == 1.0:
            FN += 1
        elif it == -0.0 and label[i] == -0.0:
            TN += 1.0
    return TP, FP, TN, FN


##ANCHOR Checks of the directory exist and if not, creates a new directory
def checkdir(directory):
    try:
        os.makedirs(directory)
    except OSError:
        pass


def cal_score(i, th, dic, pre, label):
    TP, FP, TN, FN = cal_indicator(pre, label)
    FAR, FRR = FP/(FP+TN), 1-TP/(TP+FN)
    score = FAR + FRR
    if score < dic['video'][2]+ dic['video'][3]:
        dic['video'] = [i, th, FAR, FRR]
        print(" " * 50)
        print("For the video, the result: epoch = %d, threshold = %.3f, FAR=%.4f, FRR:%.4f, Score:%4f" %(dic['video'][0], dic['video'][1], dic['video'][2], dic['video'][3], dic['video'][2]+dic['video'][3]))
    return dic['video']


##Compute the mixup data. Return mixed inputs, pairs of targets, and lambda
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

