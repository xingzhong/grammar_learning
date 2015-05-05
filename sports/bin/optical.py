import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse


feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           10, 0.03))
color = np.random.randint(0, 255, (100, 3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="input file")
    parser.parse_args()
    cap = cv2.VideoCapture(parser.parse_args().file)
    num = 1
    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(prev)
    hsv[..., 1] = 255
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, frame_gray, None, 0.5, 1, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # import ipdb
        # ipdb.set_trace()
        if num % 2 == 0:
            fig, (ax0, ax1) = plt.subplots(nrows=2)
            ax0.imshow(rgb)
            ax1.hist2d(
                flow[..., 0].ravel(), flow[..., 1].ravel(), bins=40, norm=LogNorm())
            ax1.set_aspect('equal', adjustable='box')
            # cv2.imshow('frame', rgb)
            plt.show()
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #    break
        num += 1
        prev_gray = frame_gray

if __name__ == '__main__':
    main()
