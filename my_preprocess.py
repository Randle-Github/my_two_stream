import cv2
import os
import numpy as np
import torch

'''
my_preprocess.py aims to preprocess the video into spatial and motion frames
the data will be stored in the form of the following format:

.../
    UCF101/
        frame/
            label+video_name/
                frame.jpg
        flow/
            U/
                label+video_name/
                    flow.npy
            V/
                label+video_name/
                    flow.npy
        UCF-101/
            label/
                name.avi
'''


def get_video_pic(name):  # process on one video
    cap = cv2.VideoCapture("/opt/data/private/my_two_stream/UCF101/UCF-101/" + name)
    frame_num = int(cap.get(7))  # 7 : frame number
    label, video_name = (name.rstrip(".avi")).split('/')

    os.mkdir("/opt/data/private/my_two_stream/UCF101/frame/" + label + "+" + video_name)
    os.mkdir("/opt/data/private/my_two_stream/UCF101/flow/U/" + label + "+" + video_name)
    os.mkdir("/opt/data/private/my_two_stream/UCF101/flow/V/" + label + "+" + video_name)
    interval = 5  # take 1 frame from every 5 frames
    tot = 0
    for i in range(0, frame_num - 2, int(interval)):
        cap.set(1, i)
        _, frame = cap.read()
        cap.set(1, i + 1)
        _, frame_next = cap.read()
        frame = cv2.resize(frame, (224, 224))
        frame_next = cv2.resize(frame_next, (224, 224))
        cv2.imwrite('/opt/data/private/my_two_stream/UCF101/frame/' + label + "+" + video_name + '/' + video_name + str(
            int(tot)) + '.jpg',
                    frame)  # 图片的路径

        U, V = optical_flow(frame, frame_next)
        np.save('/opt/data/private/my_two_stream/UCF101/flow/U/' + label + "+" + video_name + '/' + video_name + str(
            int(tot)) + '.npy', U)
        np.save('/opt/data/private/my_two_stream/UCF101/flow/V/' + label + "+" + video_name + '/' + video_name + str(
            int(tot)) + '.npy', V)
        tot += 1
    cap.release()


def optical_flow(prev, next):  # calculate the optical flow between prev and next
    r = cv2.calcOpticalFlowFarneback(prev[:, :, 0], next[:, :, 0], None, 0.5, 3, 15, 3, 5, 1.2, 0)
    g = cv2.calcOpticalFlowFarneback(prev[:, :, 0], next[:, :, 0], None, 0.5, 3, 15, 3, 5, 1.2, 0)
    b = cv2.calcOpticalFlowFarneback(prev[:, :, 0], next[:, :, 0], None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flowx = np.stack((r[:, :, 0], g[:, :, 0], b[:, :, 0]), axis=2)
    flowy = np.stack((r[:, :, 1], g[:, :, 1], b[:, :, 1]), axis=2)
    return flowx * 10, flowy * 10


if __name__ == "__main__":
    video_names = []
    file_name = ["testlist01", "testlist02", "testlist03", "trainlist01", "trainlist02", "trainlist03"]
    for i in range(5):
        f = open("/opt/data/private/my_two_stream/UCF_list/" + file_name[i] + ".txt", "r")
        content = f.readlines()
        f.close()
        for line in content:
            if line.rstrip() not in video_names:
                video_names.append(line.rstrip())  # label+video_name (with '.avi')
    label_list = []
    for video_name in video_names:
        get_video_pic(video_name)  # 视频的路径
    print("done!!!")
