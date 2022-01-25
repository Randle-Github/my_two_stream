import cv2
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import numpy as np
from my_utils import *


class spatial_dataset(Dataset):
    def __init__(self, root, transforms=False):
        hash, _ = read_classInd()
        self.root = root
        self.transforms = transforms
        self.video_list = []
        f = open("/opt/data/private/my_two_stream/UCF_lIST/" + root + ".txt", "r")
        content = f.readlines()
        f.close()
        for line in content:
            label, video_name = (line.rstrip('.avi')).split('/')
            self.video_list.append(
                ("/opt/data/private/my_two_stream/UCF101/frame/" + label + "+" + video_name, hash[label]))  # label+video_name (without '.avi')

    def transform(self):
        pass

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        video_site, video_label = self.video_list[index]
        frames = os.listdir(video_site)
        video = []
        for frame in frames:
            fig = torch.from_numpy(cv2.imread(video_site + '/' + frame, flags=None))
            video.append(torch.cat((fig[:, :, 0], fig[:, :, 1], fig[:, :, 2])))
        return torch.stack(video), video_label  # (3, 224, 224)


def spatial_dataloader(dataset, batch_size):
    return DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)


class motion_dataset(Dataset):
    def __init__(self, root, transforms=False):
        hash, _ = read_classInd()
        self.root = root
        self.transforms = transforms
        self.video_list = []
        f = open("/opt/data/private/my_two_stream/UCF_lIST/" + root + ".txt", "r")
        content = f.readlines()
        f.close()
        for line in content:
            label, video_name = (line.rstrip('.avi')).split('/')
            self.video_list.append(("/opt/data/private/my_two_stream/UCF101/flow/U/" + label + "+" + video_name,
                                    "/opt/data/private/my_two_stream/UCF101/flow/V/" + label + "+" + video_name,
                                    hash[label]))  # label+video_name (without '.avi')

    def transform(self):
        pass

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        flowsx = os.listdir(self.video_list[index][0])
        flowsy = os.listdir(self.video_list[index][1])
        label = self.video_list[index][2]
        video = []
        for i in range(len(flowsx)):
            flowx = np.load(self.video_list[index][0] + '/' + flowsx[i])
            flowy = np.load(self.video_list[index][1] + '/' + flowsy[i])
            flowx = torch.from_numpy(flowx)
            flowy = torch.from_numpy(flowy)
            video.append(torch.cat((torch.cat((flowx[:, :, 0], flowx[:, :, 1], flowx[:, :, 2])),
                                    torch.cat((flowy[:, :, 0], flowy[:, :, 1], flowy[:, :, 2])))))
        return torch.stack(video), label  # (6, 224, 224)


def motion_dataloader(dataset, batch_size):
    return DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)

if __name__ == "__main__":
    dataloader = spatial_dataloader()