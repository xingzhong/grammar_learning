from template import template, visM, Pts2
import cv2
import numpy as np
import sys
import time
from sklearn.neighbors import NearestNeighbors
import cPickle as pickle
from itertools import combinations
from skimage.measure import LineModel, ransac
from sklearn import linear_model

np.set_printoptions(suppress=True)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def nothing(x): pass


def initM():
    courtPts = [[613, 161], [613, 770], [394, 460], [835, 460]]  # for 0_0
    framePts = [[619, 260], [626, 720], [58, 437], [1175, 424]]
    courtPts = [[42, 161], [271, 368], [382, 162], [42, 562]] # for 25_68351
    framePts = [[386, 252], [671, 367], [933, 280], [110, 450]]
    M = cv2.getPerspectiveTransform(np.float32(courtPts), np.float32(framePts))
    return M


#@profile
def neighbors(src, dstIdx, nbrs, d=5):
    srcIdx = cv2.findNonZero(src).reshape(-1, 2)
    #srcIdx = np.vstack(np.nonzero(src)[::-1]).T
    #import ipdb; ipdb.set_trace()
    length, _ = srcIdx.shape
    rndIdx = np.random.choice(length, 600)
    srcIdx = srcIdx[rndIdx]
    distances, indices = nbrs.kneighbors(srcIdx)
    idx = distances < d
    nKeys = dstIdx[indices[idx]]
    oKeys = srcIdx[idx.ravel()]
    return oKeys, nKeys

#@profile

def ransac(oKeys, nKeys):
    def atom(o, n):
        x1, x2, y1, y2 = o[0], n[0], o[1], n[1]
        return np.array(
            [[-1, 0, x1 * x2, y1 * x2],
             [0, -1, x1 * y2, y1 * y2]]), np.array([x1-x2, y1-y2])
    n, _ = oKeys.shape
    xs = []
    for id1, id2 in combinations(range(n), 2):
        a1, y1 = atom(oKeys[id1], nKeys[id1])
        a2, y2 = atom(oKeys[id2], nKeys[id2])
        A = np.vstack(( a1, a2))
        y = np.hstack(( y1, y2))
        x, _, _, _ = np.linalg.lstsq(A, y)
        xs.append(x)
    xs = np.array(xs)
    
        

def estimate(oKeys, nKeys):
    def atom(o, n):
        x1, x2, y1, y2 = o[0], n[0], o[1], n[1]
        return np.array(
            [[-1, 0, x1 * x2, y1 * x2],
             [0, -1, x1 * y2, y1 * y2]])
    A = []
    for o, n in zip(oKeys, nKeys):
        A.append(atom(o, n))
    A = np.array(A).reshape(-1, 4).astype(float)
    y = (oKeys - nKeys).ravel()
    model = linear_model.LinearRegression(fit_intercept=False)
    model_ransac = linear_model.RANSACRegressor(model)
    model_ransac.fit(A, y)
    x = model_ransac.estimator_.coef_[0]
    #import ipdb; ipdb.set_trace()
    #x, _, _, _ = np.linalg.lstsq(A, y)
    gamma = - np.sqrt(x[1] * x[3])
    alpha = - np.sqrt(x[0] * x[2])
    f = -x[1] / gamma
    r = x[0]/(f*alpha)

    print "%.4f %.4f %.4f %.4f"%(f, r, alpha, gamma)
    return np.array([[1, 0, x[0]], [0, 1, x[1]], [x[2], x[3], 1]])


def main():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #outGrid = cv2.VideoWriter('sep/0_0.grid.avi', fourcc, 20.0, (1280, 720))
    #outFull = cv2.VideoWriter('sep/0_0.full.avi', fourcc, 20.0, (1280, 720))
    #cap = cv2.VideoCapture('sep/0_0.avi')
    #outGrid = cv2.VideoWriter('sep/25_68351.grid.avi',fourcc, 20.0, (1280,720))
    #outFull = cv2.VideoWriter('sep/25_68351.full.avi',fourcc, 20.0, (1280,720))
    cap = cv2.VideoCapture('sep/25_68351.avi')
    fn = 0
    ret, iframe = cap.read()
    H, W, _ = iframe.shape
    tpl = template()
    M = initM()

    visTpl = cv2.warpPerspective(tpl, visM(), (1280, 720))
    cRot = cv2.warpPerspective(tpl, M, (1280, 720))
    fullCourt = []
    fullImg = np.zeros_like(iframe)

    m = np.eye(3)
    tic = time.clock()
    MS = [M]
    Theta = 10
    #params = np.array([ [1.1, .9, 1.2], [.9, 1.1, 1.2], [.9, .9, 1] ])
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        fn += 1
        # if fn%2 == 0: continue
        # if fn % 6 == 0: continue
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        threshColor = cv2.inRange(
            frameHSV, np.array([0, 47, 151]), np.array([16, 255, 255]))
        threshColor = cv2.morphologyEx(
            threshColor, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        edges = cv2.Canny(threshColor, 200, 255, apertureSize=3)
        edges[565:650, 240:950] = 0
        #frame[565:650, 240:950] = 0
        cv2.circle(edges, (1042, 620), 29, 0, -1)

        dstIdx = cv2.findNonZero(edges).reshape(-1, 2)
        if len(dstIdx) < 5000:
            MS.append(MS[-1])
            continue
        nbrs = NearestNeighbors(
            n_neighbors=1, radius=1.0, algorithm='auto').fit(dstIdx)

        cnt = Theta
        converge = 100

        while cnt:
            img = frame.copy()
            cnt -= 1
            #blank = np.zeros_like(frame)

            cv2.warpPerspective(tpl, np.dot(m, M), (W, H), dst=cRot)
            # if len(nKeys) < 8000: break
            if fn  == 1 :
                oKeys, nKeys = neighbors(cRot, dstIdx, nbrs, d=10)
                dm, ret = cv2.findHomography(oKeys, nKeys, method=cv2.LMEDS)
            else:
                oKeys, nKeys = neighbors(cRot, dstIdx, nbrs, d=10)
                dm = estimate(oKeys, nKeys)
                #dm = ransac(oKeys, nKeys)
            
            print dm
            #import ipdb;
            # ipdb.set_trace()
            if dm is None:
                dm = np.eye(3)

            #converge = np.linalg.norm(dm - np.eye(3))

            # if converge < 0.45:
             #   break
            #dm = 1.2 * (dm - np.eye(3)) + np.eye(3)
            m = np.dot(dm, m)
            m = m / m[2, 2]
            img[...,1] = cv2.bitwise_or(cRot, img[...,1])
            for o, n in zip(oKeys, nKeys):
                cv2.line(img, (o[0], o[1]), (n[0], n[1]), (255,255,255), 1)
            while False:
                cv2.imshow('edges', edges)
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    return
                if key == ord('a'):
                    break
                if key == ord('c'):
                    cnt = False
                    break

        M = np.dot(m, M)
        M = M / M[2, 2]
        MS.append(M)

        alpha = np.sqrt(m[0, 2] * m[2, 0])
        gamma = np.sqrt(-m[1, 2] * m[2, 1])
        f = - m[1, 2] / gamma
        r = m[0, 2] / (alpha * f)

        # print converge
        # print 50 - cnt
        if fn > 2:
            m = .6 * (m - np.eye(3)) + np.eye(3)
        else:
            m = np.eye(3)
        img[..., 1] = cv2.bitwise_or(cRot, img[..., 1])

        inv = cv2.warpPerspective(frame, np.dot(M, np.linalg.inv(visM())),
                                  (W, H), flags=cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))
        fmask = cv2.warpPerspective(np.zeros_like(cRot),
                                    np.dot(M, np.linalg.inv(visM())),
                                    (W, H), flags=cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=255)
        #inv[...,1] = cv2.bitwise_or(visTpl, inv[...,1])

        fullT1 = cv2.bitwise_and(fullImg, fullImg, mask=fmask)

        fullImg = cv2.addWeighted(fullImg, 0.99, inv, 0.01, 0.45)
        #fullImg = inv.copy()
        fmaskI = cv2.bitwise_not(fmask)
        fullImg = cv2.bitwise_or(fullImg, fullT1)

        visImg = cv2.bitwise_and(inv, inv, mask=fmaskI)
        bg = cv2.bitwise_and(fullImg, fullImg, mask=fmask)
        visImg = cv2.add(visImg, bg)
        visImg[..., 1] = cv2.bitwise_or(visTpl, visImg[..., 1])
        toc = time.clock()

        # sys.stdout.write("\rI[%s] #%s %.4f %.4f sec/frame\n" %
        #                  (Theta - cnt, fn, converge, (toc - tic) / fn))
        # sys.stdout.write("\r%.4f %.4f %.4f %.4f" % (alpha, gamma, f, r))
        # sys.stdout.flush()
        cv2.putText(img, "[%d]#f %d %.2f %.2f sec/frame" % (Theta - cnt, fn, converge,
                                                            (toc - tic) / fn), (10, 30), FONT, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('frame', img)
        cv2.putText(visImg, "[%d]#f %d %.2f %.2f sec/frame" % (Theta - cnt, fn, converge,
                                                               (toc - tic) / fn), (10, 600), FONT, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('inv', visImg)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return
        # outGrid.write(img)
        # outFull.write(visImg)
    MS = np.array(MS)
    #pickle.dump(MS, open("sep/0_0.ms", "wb"))

if __name__ == '__main__':
    main()
