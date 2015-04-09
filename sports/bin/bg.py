import cv2
import numpy as np
import glob
from scipy.stats import mode

class SBG(object):
    def __init__(self, img, mask):
        self.m, self.n, self.k = img.shape
        self._img = None
        self._d = [[ np.empty((0,3), dtype=np.uint8) for _ in xrange(self.n)] for _ in xrange(self.m)]
        for x in xrange(self.m):
            for y in xrange(self.n):
                if mask[x][y] > 0 :
                    self._d[x][y] =np.vstack([self._d[x][y], img[x][y]])
    
    def add(self, img, mask):
        self._img = None
        for x in xrange(self.m):
            for y in xrange(self.n):
                if mask[x][y] > 0 :
                    self._d[x][y] =np.vstack([self._d[x][y], img[x][y]])
    
    def toImg(self):
        if self._img is not None:
            return self._img
        self._img = np.zeros((self.m, self.n, self.k), dtype=np.uint8)
        for x in xrange(self.m):
            for y in xrange(self.n):
                if len(self._d[x][y]) > 0 :
                    self._img[x][y] = mode(self._d[x][y], axis=0)[0]
                else:
                    self._img[x][y] = np.array([0, 0, 0],dtype=np.uint8)
        return self._img
    
    def toMotion(self):
        self._mot = np.zeros((self.m, self.n), dtype=np.uint8)
        for x in xrange(self.m):
            for y in xrange(self.n):
                if len(self._d[x][y]) > 0 :
                    model = np.mean(self._d[x][y], axis=0)
                    t = model - self._d[x][y][-1, :]
                    self._mot[x][y] = np.sqrt(np.sum(t**2))
                else:
                    self._mot[x][y] = 0
        return self._mot

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
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) > 10:
        src_pts = np.float32(
                [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
                [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 7.0)
        
    print H
    print len(kp1), len(kp2), len(good)
    
    img4 = cv2.warpPerspective(curr, H, (M, N), flags=cv2.WARP_INVERSE_MAP)
    mask4 = cv2.warpPerspective(np.zeros((720,1280), dtype=np.uint8), H, (M, N),
                                flags=cv2.WARP_INVERSE_MAP,  borderValue=255)
    mask4 = cv2.bitwise_not(mask4)
    bg.add(img4, mask4)
    cv2.bitwise_or(bgMask, mask4, dst=bgMask)
    return H

sss = "/home/xingzhong/Dropbox/dataset/f_"
frameFiles = glob.glob('/home/xingzhong/Dropbox/dataset/f_*.png')
frameFiles = sorted(frameFiles, key=lambda x:int(x[len(sss):-4]))
frames = map(lambda x: cv2.imread(x)[...,np.newaxis], frameFiles)
sX, sY = 3, 2
M, N = int(1280*sX), int(720*sY)
initM = np.array([[1.0,0, 3*1280/sX], [0,1.0, 720/sY], [0,0,1.0]])
initF = cv2.warpPerspective(frames[0][...,0], initM, (M, N))
mask = cv2.warpPerspective(np.zeros((720,1280), dtype=np.uint8), initM, (M, N), borderValue=255)
mask = cv2.bitwise_not(mask)
bg = SBG(initF, mask)
for idx, frame in enumerate(frames[1:]):
    f2f(bg, frame[...,0], mask)
    cv2.imwrite('/home/xingzhong/Pictures/c_%s.png'%idx, bg.toImg())
    cv2.imwrite('/home/xingzhong/Pictures/m_%s.png'%idx, bg.toMotion())
