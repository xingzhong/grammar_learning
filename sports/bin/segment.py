import numpy as np
from collections import deque
import cv2
import cPickle as pickle
import pandas as pd
import csv
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from skimage.util.shape import view_as_blocks

FONT = cv2.FONT_HERSHEY_SIMPLEX


def getFrame(vf, frames):
    cap = cv2.VideoCapture(vf)
    num = 0
    results = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if num > frames[-1]:
            break
        if num % 5000 == 0:
            print num
        if num in frames:
            results.append((frame, num))
        num += 1
    return results


def sepVideo(vf, idx):
    cap = cv2.VideoCapture(vf)
    num = 0
    flag = False
    nv = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        'sep/%s_%s.avi' % (nv, num), fourcc, 20.0, (1280, 720))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if num > len(idx):
            break
        if num % 1000 == 0:
            print num
        if idx[num]:
            flag = True
            out.write(frame)
        if (not idx[num]) and flag:
            flag = False
            nv += 1
            out.release()
            print "new video %s_%s.avi" % (nv, num)
            out = cv2.VideoWriter(
                'sep/%s_%s.avi' % (nv, num), fourcc, 20.0, (1280, 720))
        num += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def globalMask(frame):
    blank = np.zeros_like(frame)
    blank[180:, :] = 255
    blank[550:680, 240:955] = 0
    return blank


def getFeature():
    cap = cv2.VideoCapture('/home/xingzhong/Videos/heat.mkv')
    fn = 0
    ret, iframe = cap.read()
    H, W, _ = iframe.shape
    gmask = globalMask(iframe)
    nznsLeft = deque(maxlen=25)
    nznsCenter = deque(maxlen=25)
    nznsRight = deque(maxlen=25)
    features = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        # if fn >= 100000: break
        fn += 1

        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ret, threshColor = cv2.threshold(
            frameHSV[:, :, 0], 30, 179, cv2.THRESH_BINARY_INV)
        #threshColorMask = cv2.bitwise_and(frame, frame, mask = threshColor)
        #import ipdb; ipdb.set_trace()
        #imgs = view_as_blocks(threshColorMask, block_shape = (H/9, W/16, 3)).reshape(-1, H/9, W/16, 3)
        imgs = view_as_blocks(
            threshColor, block_shape=(H / 9, W / 16)).reshape(-1, H / 9, W / 16)
        nzns = map(lambda x: np.count_nonzero(x) / float(x.size), imgs)
        #import ipdb; ipdb.set_trace()
        #nznLeft  = 3 * np.count_nonzero(threshColorMask[:, :W/3]) / float(threshColorMask.size)
        #nznCenter = 3 * np.count_nonzero(threshColorMask[:, W/3:2*W/3]) / float(threshColorMask.size)
        #nznRight = 3 * np.count_nonzero(threshColorMask[:, -W/3:]) / float(threshColorMask.size)
            # nznsLeft.append(nznLeft)
            # nznsCenter.append(nznCenter)
            # nznsRight.append(nznRight)
        features.append(nzns)
        if fn % 100 == 0:
            cv2.putText(frame, "#f %d" %
                        fn, (10, 30), FONT, 1, (255, 255, 255), 1, cv2.LINE_AA)
            #cv2.putText(threshColorMask, "%d"%(int(100*nznLeft)) ,(W/6, 100), FONT, 1,(255,255,255),1,cv2.LINE_AA)
            #cv2.putText(threshColorMask, "%d"%(int(100*nznCenter)) ,(W/2, 100), FONT, 1,(255,255,255),1,cv2.LINE_AA)
            #cv2.putText(threshColorMask, "%d"%(int(100*nznRight)) ,(5*W/6, 100), FONT, 1,(255,255,255),1,cv2.LINE_AA)
            mask = np.array(nzns).reshape(9, 16)
            cv2.imshow('mask', mask)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    pickle.dump(features, open("segment_features_full.p", "wb"))
    cap.release()
    cv2.destroyAllWindows()


def learnDetector():

    features = pickle.load(open("segment_features_full.p", "rb"))
    features = np.array(features)
    truth = np.zeros(40000)
    with open('segment_truth_40000.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for i, j in reader:
            truth[int(i):int(j)] = 1

    #mod = svm.LinearSVC(C=1.0).fit(features, truth)
    #mod = GradientBoostingClassifier().fit(features[:40000, :], truth)
    mod = LogisticRegression().fit(features[:40000, :], truth)
    #err = np.where(mod.predict(features) != truth)[0]
    # for e in getFrame("/home/xingzhong/Videos/heat.mkv", err):
    #	cv2.imwrite("err/%s.png"%e[1],e[0])
    guess = mod.predict_proba(features)[:, -1]
    guess_smooth = np.convolve(guess, np.ones(250) / 250)

    #import ipdb; ipdb.set_trace()
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(guess_smooth > 0.75)
    ax.plot(truth)
    ax.set_ylim(-0.1, 1.1)
    plt.show()
    sepVideo("/home/xingzhong/Videos/heat.mkv", guess_smooth > 0.75)
    #coef = mod.coef_.reshape(9, 16)
    #plt.imshow(mod); plt.show()
    #import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    # getFeature()
    learnDetector()
