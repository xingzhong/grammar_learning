import cv2
import numpy as np
import glob
import argparse


class SBG(object):
    def __init__(self, img, mask):
        self.m, self.n, self.k = img.shape
        self._img = None
        self._d = [[ np.empty((0,3), dtype=np.uint8) for _ in xrange(self.n)] for _ in xrange(self.m)]
        for x in xrange(self.m):
            for y in xrange(self.n):
                if mask[x][y] > 0:
                    self._d[x][y] = np.vstack([self._d[x][y], img[x][y]])

    def add(self, img, mask):
        self._img = None
        for x in xrange(self.m):
            for y in xrange(self.n):
                if mask[x][y] > 0:
                    self._d[x][y] = np.vstack([self._d[x][y], img[x][y]])

    def toImg(self):
        if self._img is not None:
            return self._img
        self._img = np.zeros((self.m, self.n, self.k), dtype=np.uint8)
        for x in xrange(self.m):
            for y in xrange(self.n):
                if len(self._d[x][y]) > 0:
                    self._img[x][y] = np.mean(self._d[x][y], axis=0)
                else:
                    self._img[x][y] = np.array([0, 0, 0], dtype=np.uint8)
        return self._img

    def toMotion(self):
        self._mot = np.zeros((self.m, self.n), dtype=np.uint8)
        for x in xrange(self.m):
            for y in xrange(self.n):
                if len(self._d[x][y]) > 0:
                    model = np.mean(self._d[x][y], axis=0)
                    t = model - self._d[x][y][-1, :]
                    self._mot[x][y] = np.sqrt(np.sum(t**2))
                else:
                    self._mot[x][y] = 0
        return self._mot


def fastFeature(bg, curr, bgMask):
    orb = cv2.ORB()
    prev = bg.toImg()
    kp1, des1 = orb.detectAndCompute(prev, bgMask)
    kp2, des2 = orb.detectAndCompute(curr, None)
    img = cv2.drawKeypoints(bg, kp2, color=(255, 0, 0))
    cv2.imshow('fast', img)

def f2f(bg, curr, bgMask):
    sift = cv2.SIFT(nOctaveLayers=3, contrastThreshold=0.05, edgeThreshold=10)
    prev = bg.toImg()
    N, M, _ = prev.shape
    kp1, des1 = sift.detectAndCompute(prev, bgMask)
    kp2, des2 = sift.detectAndCompute(curr, None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > 10:
        src_pts = np.float32(
                [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
                [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
        matchesMask = mask.ravel().tolist()

    print H
    print len(kp1), len(kp2), len(good)
    
    img4 = cv2.warpPerspective(curr, H, (M, N), flags=cv2.WARP_INVERSE_MAP)
    mask4 = cv2.warpPerspective(np.zeros((720,1280), dtype=np.uint8), H, (M, N),
                                flags=cv2.WARP_INVERSE_MAP,  borderValue=255)
    mask4 = cv2.bitwise_not(mask4)
    bg.add(img4, mask4)
    cv2.bitwise_or(bgMask, mask4, dst=bgMask)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 0)

    img3 = cv2.drawMatches(prev,kp1,curr,kp2,good,None,**draw_params)
    cv2.imshow('sift', img3)

    return H


def main():
    sss = '/home/xingzhong/Dropbox/dataset/f_'
    frameFiles = glob.glob('/home/xingzhong/Dropbox/dataset/f_*.png')
    frameFiles = sorted(frameFiles, key=lambda x: int(x[len(sss):-4]))
    frames = map(lambda x: cv2.imread(x)[..., np.newaxis], frameFiles)
    scale = 3
    M, N = int(1280 * scale), int(720 * scale)
    initM = np.array(
        [[1.0, 0, 3 * 1280 / scale], [0, 1.0, 3 * 720 / scale], [0, 0, 1.0]])
    initF = cv2.warpPerspective(frames[0][..., 0], initM, (M, N))
    mask = cv2.warpPerspective(
        np.zeros((720, 1280), dtype=np.uint8), initM, (M, N), borderValue=255)
    mask = cv2.bitwise_not(mask)

    bg = SBG(initF, mask)

    for idx, frame in enumerate(frames[1:]):
        f2f(bg, frame[..., 0], mask)
        cv2.imwrite('/home/xingzhong/Dropbox/dataset/c_%s.png' %
                    idx, bg.toImg())
        cv2.imwrite('/home/xingzhong/Dropbox/dataset/m_%s.png' %
                    idx, bg.toMotion())


def build():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="input file")
    parser.parse_args()
    cap = cv2.VideoCapture(parser.parse_args().file)
    num = 1
    n = 0
    scale = 1
    for _ in range(10):
        ret, iframe = cap.read()
    M, N = int(1280 * scale), int(720 * scale)
    initM = np.array(
        [[1.0, 0, 1], [0, 1.0, 1], [0, 0, 1.0]])
    initF = cv2.warpPerspective(iframe, initM, (M, N))
    mask = cv2.warpPerspective(
        np.zeros((720, 1280), dtype=np.uint8), initM, (M, N), borderValue=255)
    mask = cv2.bitwise_not(mask)
    bg = SBG(initF, mask)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if num % 10 == 0:
            f2f(bg, frame, mask)
            fastFeature(bg, frame, mask)
            cv2.imshow('bg', bg.toImg())
        cv2.imshow('frame', frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
        num += 1

if __name__ == '__main__':
    #main()
    build()
