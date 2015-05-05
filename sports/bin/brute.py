import cv2
import numpy as np
import pandas as pd
import cPickle as pickle
from matplotlib import pyplot as plt
from template import template
np.set_printoptions(suppress=True)


def vis(ms, nms):
    msdf = pd.DataFrame(ms.reshape(-1, 9))
    nmsdf = pd.DataFrame(ms.reshape(-1, 9))
    mtxRs = np.array(map(lambda x: cv2.RQDecomp3x3(x)[1], ms))
    mtxQs = np.array(map(lambda x: cv2.RQDecomp3x3(x)[2], ms))
    mtxRsDf = pd.DataFrame(mtxRs.reshape(-1, 9))
    mtxQsDf = pd.DataFrame(mtxQs.reshape(-1, 9))

    #import ipdb; ipdb.set_trace()
    msdf.plot(subplots=True, layout=(3, 3))
    mtxRsDf.plot(subplots=True, layout=(3, 3))
    mtxQsDf.plot(subplots=True, layout=(3, 3))
    msdf.hist(layout=(3, 3))
    plt.show()
    #import ipdb; ipdb.set_trace()


def rollingMean(ms):

    pm = pd.DataFrame(ms.reshape(-1, 9))
    nms = pd.rolling_mean(
        pm, 3, min_periods=1, center=True).values.reshape(-1, 3, 3)
    mtxRs = np.array(map(lambda x: cv2.RQDecomp3x3(x)[1], ms))
    mtxQs = np.array(map(lambda x: cv2.RQDecomp3x3(x)[2], ms))
    nms = []
    n, _, _ = mtxRs.shape
    params = []
    for i in range(n):
        r = mtxRs[i]
        q = mtxQs[i]
        theta = -np.arcsin(q[0, 1])
        dx = r[0, 2]
        dy = r[1, 2]
        focus = r[1, 1]
        shear = r[0, 1]
        ratio = r[0, 0] / focus
        params.append([focus, shear, dx, theta, ratio, dy, q[2, 0], q[2, 1]])

    df = pd.DataFrame(
        params, columns=['focus', 'shear', 'dx', 'theta', 'ratio', 'dy', 'p1', 'p2'])
    nmsdf = pd.rolling_mean(df, 10, min_periods=1, center=True).values
    nms = []
    for focus, shear, dx, theta, ratio, dy, p1, p2 in df.values:
        r = np.array([[ratio * focus, shear, dx], [0, focus, dy], [0, 0, 1]])
        q = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0], [p1, p2, 1]])
        nms.append(np.dot(r, q))

    df.plot(subplots=True, layout=(3, 3))
    #nms.plot(subplots=True, layout=(3, 3))
    plt.show()
    return nms


def play():
    cap = cv2.VideoCapture('sep/25_68351.avi')
    ms = pickle.load(open("sep/25_68351.ms", "rb"))
    nms = rollingMean(ms)
    #vis(ms, nms)
    fn = 0
    ret, iframe = cap.read()
    H, W, _ = iframe.shape
    tpl = template()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        fn += 1
        res, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(ms[fn])
        # print mtxR
        # print mtxQ
        import ipdb
        ipdb.set_trace()
        crotMean = cv2.warpPerspective(tpl, nms[fn], (W, H))
        frame[..., 1] = cv2.bitwise_or(crotMean, frame[..., 1])
        cv2.imshow('frame', frame)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            return
if __name__ == '__main__':

    play()
