#!/usr/bin/env python

'''
Lucas-Kanade homography tracker
===============================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames. Finds homography between reference and current views.

Usage
-----
lk_homography.py [<video_source>]


Keys
----
ESC   - exit
SPACE - start tracking
r     - toggle RANSAC
'''

import numpy as np
import cv2
import video
from common import draw_str
import numpy.ma as ma

class SBG(object):
    def __init__(self, img, mask):
        self.m, self.n, self.k = img.shape
        self._img = None
        #import ipdb; ipdb.set_trace()
        self._d = img[:, :, np.newaxis, :]
        self._mask = mask[..., np.newaxis] > 0
    
    def add(self, img, mask):
        self._img = None
        self._d = np.concatenate((self._d, img[:, :, np.newaxis, :]), axis=2)
        self._mask = np.concatenate((self._mask, mask[..., np.newaxis]>0), axis=2)
        #import ipdb; ipdb.set_trace()
        
    
    def toImg(self):
        if self._img is not None:
            return self._img
        mask = np.concatenate([self._mask[...,np.newaxis]]*3, axis=3)
        data = ma.array(self._d, mask = mask)
        self._img = np.mean(data, axis=2).filled(128).astype(np.uint8)
        return self._img

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 3500,
                       qualityLevel = 0.01,
                       minDistance = 10,
                       blockSize = 5 )

def checkedTrace(img0, img1, p0, back_threshold = 1.0):
    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    status = d < back_threshold
    return p1, status

green = (0, 255, 0)
red = (0, 0, 255)

class App:
    def __init__(self, video_src):
        self.cam = video.create_capture(video_src)
        self.p0 = None
        self.use_ransac = True
        self.frame_idx = 0
        self.detect_interval = 1
        self.init_interval = 50
        self.incrH = np.eye(3, dtype=np.float32)
        self.incrH[0, 0] = 2.0
        self.incrH[1, 1] = 2.0
        self.incrH = np.array(
            [[1.8, 0, -500.], [0, 1.8, -300.], [0, 0, 1]])
        #self.incrH = np.array(
        #    [[4, 0, -3000.], [0, 4, -600.], [0, 0, 1]])
        self.blank = None
        self.mask = None
        self.bg = None
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('/tmp/test.avi', fourcc, 20.0, (1280, 720))

    def run(self):
        while True:
            ret, frame = self.cam.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()
            if self.p0 is not None:
                p2, trace_status = checkedTrace(self.gray1, frame_gray, self.p1)

                self.p1 = p2[trace_status].copy()
                self.p0 = self.p0[trace_status].copy()
                self.gray1 = frame_gray

                if len(self.p0) < 4:
                    self.p0 = None
                    continue
                H, status = cv2.findHomography(self.p0, self.p1, (0, cv2.RANSAC)[self.use_ransac], 1.5)
                if H is None:
                    continue
                
                self.incrH = np.dot(H, self.incrH)
                self.incrH = self.incrH / self.incrH[2,2]
                print self.incrH
                h, w = frame.shape[:2]
                overlay = cv2.warpPerspective(self.frame0, H, (w, h))
                vis = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0.0)
                if self.blank is None:
                    self.blank = np.zeros((h, w)).astype(np.uint8)
                proj = cv2.warpPerspective(frame, self.incrH, (w, h),
                            flags=cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0))
                
                xor = cv2.bitwise_xor(vis, overlay)
                xor = cv2.cvtColor(xor, cv2.COLOR_BGR2GRAY)
                xor = cv2.GaussianBlur(xor,(5,5),0)
                ret, xor = cv2.threshold(xor, 20, 255, cv2.THRESH_BINARY)
                #xor = cv2.morphologyEx(xor, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))
                xor = cv2.morphologyEx(xor, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
                xor = cv2.morphologyEx(xor, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
                #xor = cv2.GaussianBlur(xor,(5,5),0)
                xor_inv = cv2.bitwise_not(xor)
                mask = cv2.warpPerspective(xor, self.incrH, (w, h),
                            flags=cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=255)
                #fmask = cv2.bitwise_not(mask)
                #if self.mask is None:
                #    self.mask = fmask
                #else:
                #    self.mask = cv2.bitwise_or(self.mask, fmask)
                if self.bg is None:
                    #pass
                    #self.bg = proj.copy()
                    self.bg = SBG(proj, mask)
                else:
                    #T = cv2.bitwise_or(self.bg, proj, mask=mask)
                    #bg = cv2.bitwise_and(proj, proj, mask=fmask)
                    #self.bg = cv2.bitwise_or(bg, T)
                    self.bg.add(proj, mask)
                    pass
                forgrd = cv2.bitwise_and(frame, frame, mask=xor)
                greenbg = np.zeros_like(frame)
                greenbg[...,1] = 255
                forgrd2 = cv2.bitwise_and(greenbg, greenbg, mask=xor_inv)
                forgrd = forgrd + forgrd2
                #import ipdb; ipdb.set_trace()
                cv2.imshow('proj', proj)
                cv2.imshow('xor', xor)
                cv2.imshow('forgrd', forgrd)
                self.out.write(forgrd)
                for (x0, y0), (x1, y1), good in zip(self.p0[:,0], self.p1[:,0], status[:,0]):
                    if good:
                        cv2.line(vis, (x0, y0), (x1, y1), (0, 255, 0))
                    cv2.circle(vis, (x1, y1), 2, (red, green)[good], -1)
                draw_str(vis, (20, 20), 'track count: %d' % len(self.p1))
                if self.use_ransac:
                    draw_str(vis, (20, 40), 'RANSAC')
            else:
                p = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
                if p is not None:
                    for x, y in p[:,0]:
                        cv2.circle(vis, (x, y), 2, green, -1)
                    draw_str(vis, (20, 20), 'feature count: %d' % len(p))

            self.frame_idx += 1
            cv2.imshow('lk_homography', vis)

            ch = 0xFF & cv2.waitKey(10)
            if ch == 27:
                break
            if ch == ord(' ') or (self.frame_idx % self.detect_interval == 0):
                
                self.frame0 = frame.copy()
                self.p0 = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
                if self.p0 is not None:
                    self.p1 = self.p0
                    self.gray0 = frame_gray
                    self.gray1 = frame_gray
                    
            if ch == ord('r'):
                self.use_ransac = not self.use_ransac
            if self.frame_idx % self.init_interval == 0:
                if self.bg is not None:
                    cv2.imshow('bg', self.bg.toImg())
        cv2.imwrite('/tmp/bg.png', self.bg.toImg())


def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print __doc__
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
