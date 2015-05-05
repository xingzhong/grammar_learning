import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

def neighbors(src, dst):
	#d = 10
	srcIdx = np.vstack(np.nonzero(src)[::-1]).T
	dstIdx = np.vstack(np.nonzero(dst)[::-1]).T
	length, _ = dstIdx.shape
	rndIdx = np.random.choice(length, 10000)
	dstIdx = dstIdx[rndIdx]
	nbrs = NearestNeighbors(n_neighbors=1,  algorithm = 'ball_tree').fit(srcIdx)
	distances, indices = nbrs.kneighbors(dstIdx)

	nKeys = srcIdx[indices]
	return nKeys, dstIdx.reshape(-1, 1, 2)


def multiRansac(oKeys, nKeys):
	h = 0
	length, _, _ = oKeys.shape
	while h < 1000:
		rndIdx = np.random.choice(length, 4)
		k1, k2 = oKeys[rndIdx], nKeys[rndIdx]
		h1 = cv2.findHomography(k1,k2)
		h2 = cv2.findHomography(k2,k1)
		import ipdb; ipdb.set_trace()

def main():
	img1 = cv2.imread('test1_30f.png', 0)
	img2 = cv2.imread('test2_60f.png', 0)
	kp1 = cv2.cornerHarris(img1,2,3,0.04)
	kp2 = cv2.cornerHarris(img2,2,3,0.04)
	blank = np.zeros((720, 1280, 3))
	blank[...,0] = cv2.Canny(img1,200,255,apertureSize = 3)
	blank[...,2] = cv2.Canny(img2,200,255,apertureSize = 3)

	oKeys, nKeys = neighbors(blank[...,0], blank[..., 2])
	print oKeys.shape
	for idx, pts in enumerate(np.dstack((oKeys, nKeys))):
		x1,y1,x2,y2 = pts[0]
		cv2.line( blank, (x1,y1), (x2, y2), (0,255,0), 1)

	multiRansac(oKeys, nKeys)
	#blank[kp2>0.01*kp2.max()]=[0,0,255]
	#blank[kp1>0.01*kp1.max()]=[255,0,0]
	cv2.imshow('dst',blank)
	if cv2.waitKey(0) & 0xff == 27:
		cv2.destroyAllWindows()

if __name__ == '__main__':
	main()