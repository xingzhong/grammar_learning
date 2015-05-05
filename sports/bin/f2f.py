# frame 2 frame

from template import template, visM, Pts2
import cv2
import numpy as np
import sys
import time
from sklearn.neighbors import NearestNeighbors
import cPickle as pickle
from template import visM

np.set_printoptions(suppress=True)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def nothing(x): pass


def initM():
    courtPts = [[613, 161], [613, 770], [394, 460], [835, 460]]  # for 0_0
    framePts = [[619, 260], [626, 720], [58, 437], [1175, 424]]
    # courtPts = [[42, 161], [271, 368], [382, 162], [42, 562]] # for 25_68351
    #framePts = [[386, 252], [671, 367], [933, 280], [110, 450]]
    M = cv2.getPerspectiveTransform(np.float32(courtPts), np.float32(framePts))
    return M


def neighbors(src, dstIdx, nbrs, d=5):
    srcIdx = cv2.findNonZero(src).reshape(-1, 2)
    # srcIdx = np.vstack(np.nonzero(src)[::-1]).T
    # import ipdb; ipdb.set_trace()
    length, _ = srcIdx.shape
    rndIdx = np.random.choice(length, 600)
    srcIdx = srcIdx[rndIdx]
    distances, indices = nbrs.kneighbors(srcIdx)
    idx = distances < d
    nKeys = dstIdx[indices[idx]]
    oKeys = srcIdx[idx.ravel()]
    return oKeys, nKeys


def edgeImg(frame):
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    threshColor = cv2.inRange(
        frameHSV, np.array([0, 47, 151]), np.array([16, 255, 255]))
    threshColor = cv2.morphologyEx(
        threshColor, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    edges = cv2.Canny(threshColor, 200, 255, apertureSize=3)
    edges[565:650, 240:950] = 0
    cv2.circle(edges, (1042, 620), 29, 0, -1)
    return edges


def f2fEst(prev, curr):
    prevPts = cv2.findNonZero(prev).reshape(-1, 2)
    currPts = cv2.findNonZero(curr).reshape(-1, 2)

    nbrs = NearestNeighbors(
        n_neighbors=1, radius=1.0, algorithm='auto').fit(prevPts)
    length, _ = currPts.shape
    rndIdx = np.random.choice(length, 1000)
    currPts = currPts[rndIdx]
    distances, indices = nbrs.kneighbors(currPts)
    idx = distances < 50
    oKeys = prevPts[indices[idx]]
    nKeys = currPts[idx.ravel()]
    print len(prevPts), len(currPts), len(nKeys)
    M, mask = cv2.findHomography(oKeys.reshape(-1, 1, 2),
                                 nKeys.reshape(-1, 1, 2),
                                 cv2.RANSAC, 5.0)

    if mask is not None and np.sum(mask) > 200:
        matchesMask = mask.ravel().tolist()
        return oKeys, nKeys, matchesMask, M
    else:
        return [], [], [], None


def mix(curr, bg):
        # given bg, accumulate curr
    pass


def siftFeature(prev, curr, mask):
    sift = cv2.SIFT(nOctaveLayers=3, contrastThreshold=0.05, edgeThreshold=10)
    #surf = cv2.SURF(1000)
    kp1, des1 = sift.detectAndCompute(prev, mask)
    kp2, des2 = sift.detectAndCompute(curr, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []

    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append(m)
    print len(kp1), len(kp2), len(matches), len(good)
    if len(good) > 10:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        if M is None:
            return None, None
        matchesMask = mask.ravel().tolist()
        img = curr.copy()
        for idx, m in enumerate(matchesMask):
            x1, y1 = map(int, src_pts[idx][0])
            x2, y2 = map(int, dst_pts[idx][0])
            if m == 1:

                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.circle(img, (x1, y1), 2, (0, 0, 255), -1)
                cv2.circle(img, (x2, y2), 1, (0, 0, 255), -1)
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=2)
        img3 = cv2.drawMatches(prev,
                               kp1, curr, kp2, good, None, **draw_params)

        cv2.imshow('sift', img3)
    else:
        print "Not enough matches are found - %d/%d" % (len(good), 50)
        matchesMask = None
        M = None
        img = curr.copy()

    return img, M


def main():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # outGrid = cv2.VideoWriter('sep/0_0.grid.avi', fourcc, 20.0, (1280, 720))
    outFull = cv2.VideoWriter('sep/full.avi', fourcc, 20.0, (1280, 720))
    cap = cv2.VideoCapture('sep/0_0.avi')
    # outGrid = cv2.VideoWriter('sep/25_68351.grid.avi',fourcc, 20.0, (1280,720))
    # outFull = cv2.VideoWriter('sep/25_68351.full.avi',fourcc, 20.0, (1280,720))
    # cap = cv2.VideoCapture('sep/25_68351.avi')
    fn = 0
    ret, iframe = cap.read()
    H, W, _ = iframe.shape
    M = np.dot(initM(), np.linalg.inv(visM()))
    M = np.array(
        [[4., 0, -2000.], [0, 4., - 1200.], [0, 0, 1]])
    #M = np.eye(3)
    bg = np.zeros_like(iframe)

    prev = cv2.warpPerspective(iframe, M, (W, H),
                               flags=cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))
    # prev = cv2.Canny(iframe, 100, 255, apertureSize=3)
    # prev[565:650, 240:950] = 0
    # cv2.circle(prev, (1042, 620), 29, 0, -1)
    # params = np.array([ [1.1, .9, 1.2], [.9, 1.1, 1.2], [.9, .9, 1] ])
    blank = np.zeros((H, W)).astype(np.uint8)
    bgMask = cv2.warpPerspective(blank, M, (W, H),
                                 flags=cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=255)
    bgMask = cv2.bitwise_not(bgMask)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        fn += 1
        if fn % 30 != 0:
            continue

        img, dm = siftFeature(prev, frame, bgMask)
        if dm is None:
            continue
        M = np.dot(dm, M)
        M = M / M[2, 2]
        print M
        proj = cv2.warpPerspective(frame, dm, (W, H),
                                   flags=cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(blank, dm, (W, H),
                                   flags=cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=255)
        fmask = cv2.bitwise_not(mask)
        bgMask = cv2.bitwise_or(bgMask, fmask)
        if fn == 1:
            bg = proj.copy()
        else:
            T = cv2.bitwise_or(bg, proj, mask=mask)
            bg = cv2.bitwise_and(proj, proj, mask=fmask)
            bg = cv2.bitwise_or(bg, T)
        prev = bg
        while False:
            cv2.imshow('img', img)
            cv2.imshow('proj', proj)
            cv2.imshow('mask', bgMask)
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                return
            if key == ord('a'):
                break
            if key == ord('c'):
                break
        cv2.imshow('proj', proj)
        cv2.imshow('mask', bgMask)
        cv2.imshow('bg', bg)
        outFull.write(bg)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            return
if __name__ == '__main__':
    main()
