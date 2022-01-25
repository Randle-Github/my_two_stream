import numpy as np
import os
from my_network import *
from my_utils import *
from my_dataloader import *

import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Spatial_CNN():
    def __init__(self, lr):
        self.lr = lr
        self.model = resnet101(pretrained=True, channel=3).cuda()
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1, verbose=True)

    def forward(self, x):  # (n, 224, 224, 3)
        return self.model.forward(x)

    def train(self, dataloader):
        for i, (video, label) in enumerate(dataloader):  # (n,3,224,224)
            batch_size = video.size[0]
            video = torch.unsqueeze(video, 0)
            video[0] = batch_size
            video = Variable(video)
            target = torch.zeros((batch_size, 101))
            for i in range(batch_size):
                target[i][int(label[i])] = 1
            target = Variable(target)

            logits = self.forward(video)
            pred = nn.Softmax(logits)
            loss = self.criterion(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def test(self, dataloader):
        pred = torch.zeros((len(dataloader.dataset), 101))
        target = torch.zeros(len(dataloader.dataset))
        for i, (video, label) in enumerate(dataloader):  # (n,3,224,224)
            batch_size = video.size[0]
            video = torch.unsqueeze(video, 0)
            video[0] = batch_size
            logits = self.forward(video)
            pred[i] = nn.Softmax(logits)
            target[i] = label
        return pred, target # (n,101), (n)

if __name__ == '__main__':
    model = Spatial_CNN(0.01)
    file_name = ["testlist01", "testlist02", "testlist03", "trainlist01", "trainlist02", "trainlist03"]
    dataloader1 = spatial_dataloader(file_name[0], 64)
    dataloader2 = spatial_dataloader(file_name[1], 64)
    dataloader3 = spatial_dataloader(file_name[2], 64)
    dataloader4 = spatial_dataloader(file_name[3], 1)
    dataloader5 = spatial_dataloader(file_name[4], 1)
    dataloader6 = spatial_dataloader(file_name[5], 1)
    for i in range(1000):
        model.train(dataloader1)
        model.train(dataloader2)
        model.train(dataloader3)
        acc = 0
        acc += top_1_acc(model.test(dataloader4))
        acc += top_1_acc(model.test(dataloader5))
        acc += top_1_acc(model.test(dataloader6))
        print("epoch{}: accuracy = {}%".format(i, acc*100))
    pred4, _ = model.test(dataloader4)
    pred5, _ = model.test(dataloader5)
    pred6, _ = model.test(dataloader6)
    np.save("/opt/data/private/my_two_stream/testlist01_spatial_pred.npy", np.array(pred4))
    np.save("/opt/data/private/my_two_stream/testlist02_spatial_pred.npy", np.array(pred5))
    np.save("/opt/data/private/my_two_stream/testlist03_spatial_pred.npy", np.array(pred6))

