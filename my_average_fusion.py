import os
from my_utils import *
import numpy as np

if __name__ == '__main__':
    file_name = ["testlist01", "testlist02", "testlist03", "trainlist01", "trainlist02", "trainlist03"]
    spatial_pred = np.load()
    motion_pred = np.load()
    fusion_pred = (spatial_pred + motion_pred) / 2
    fus_pred = np.argmax(fusion_pred, axis=0)
