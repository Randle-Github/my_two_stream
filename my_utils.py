import numpy as np
import os
import torch


def read_classInd():
    # f = open("/opt/data/private/my_two_stream/UCF_list/classInd.txt", "w")
    f = open("UCF_list/classInd.txt", "w")
    content = f.readlines()
    f.close()
    hash = {}
    dehash = {}
    for line in content:
        num, name = line.split(" ")
        hash[name] = num
        dehash[num] = name
    return hash, dehash


def top_1_acc(pred, label):
    batch_size = pred.size[0]
    acc = 0
    for i in range(batch_size):
        if torch.argmax(pred[i]) == int(label[i]):
            acc += 1
    return acc / batch_size


def top_5_acc(pred, label):
    pass
