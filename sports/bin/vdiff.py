import numpy as np
import cv2
from template import template, POI
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import cStringIO

FONT = cv2.FONT_HERSHEY_SIMPLEX

def neighbors(src, dst):
	#d = 10
	srcIdx = np.vstack(np.nonzero(src)[::-1]).T
	dstIdx = np.vstack(np.nonzero(dst)[::-1]).T
	nbrs = NearestNeighbors(n_neighbors=1,  algorithm = 'ball_tree').fit(srcIdx)

	#length, _ = srcIdx.shape
	#rndIdx = np.random.choice(length, 3000)
	#srcIdx = srcIdx[rndIdx]

	distances, indices = nbrs.kneighbors(dstIdx)
	
	nKeys = srcIdx[indices]
	return nKeys, dstIdx.reshape(-1, 1, 2)
	#idx = distances<d
	#nKeys = dstIdx[indices[idx]]
	#oKeys = srcIdx[idx.ravel()]
	return oKeys, nKeys

def main():
	cap = cv2.VideoCapture('sep/0_0.avi')
	fn = 0

	# FLANN parameters
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary
	flann = cv2.FlannBasedMatcher(index_params,search_params)
	sift = cv2.SIFT()


	ret, iframe = cap.read()
	igray= cv2.cvtColor(iframe,cv2.COLOR_BGR2GRAY)
	iedges = cv2.Canny(igray,200,255,apertureSize = 3)
	iedges[550:680, 240:955] = 0

	#kp1,des1 = sift.detectAndCompute(igray,None)
	kp1 = cv2.cornerHarris(igray,2,3,0.04)

	H, W, _ = iframe.shape
	tpl = template()
	blank = np.zeros_like(iframe)
	blank[...,0] = iedges
	prev = igray
	est = np.zeros_like(igray)
	m = np.eye(3)
	while(cap.isOpened()):
		ret, frame = cap.read()
		if not ret: break
		fn += 1	
		gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray,200,255,apertureSize = 3)
		edges[550:680, 240:955] = 0
		#kp2,des2 = sift.detectAndCompute(gray,None)
		blank[...,2] = edges
		kp2 = cv2.cornerHarris(gray,2,3,0.04)
		
		#matches = flann.knnMatch(des1,des2,k=2)

		
		#import ipdb; ipdb.set_trace()
		
		
		#blank = np.zeros_like(frame)
		#blank[kp2>0.01*kp2.max()]=[0,0,255]
		#blank[kp1>0.01*kp1.max()]=[255,0,0]
		oKeys, nKeys = neighbors(blank[...,0], blank[..., 2])
		#dm, ret = cv2.findHomography(oKeys, nKeys, method=cv2.RANSAC, ransacReprojThreshold=5)
		dm = None
		if dm is not None:
			blank = np.zeros_like(frame)
			m = np.dot(dm, m)
			cv2.warpPerspective(igray, m, (W, H), dst = est)

			for idx, pts in enumerate(np.dstack((oKeys, nKeys))):
				x1,y1,x2,y2 = pts[0]
				if ret[idx][0] == 1:
					blank[y2, x2] = (0,0,255)
				else:
					blank[y2, x2] = (255,0,0)

		#diff = (nKeys - oKeys).reshape(-1, 2)
		#fig, ax = plt.subplots()
		#ax.scatter( diff[:, 0], diff[:, 1])
		#ax.set_xlim([-50, 50])
		#ax.set_ylim([-50, 50])
		#ax.grid()
		#fig.canvas.draw()
		#test = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
		#test = test.reshape(fig.canvas.get_width_height()[::-1] + (3,))
		#import ipdb; ipdb.set_trace()
		
		#blank = cv2.drawKeypoints(blank, kp1, color=(255,0,0))
		#blank = cv2.drawKeypoints(blank, kp2, color=(0,0,255))
		#for x, y in diff:
		#	cv2.circle( blank, (int(10*(x+50)), int(10*(y+50))), 4,  (0,0,255), 2)
		
		#for i,(m,n) in enumerate(matches):
		#	if m.distance < 0.7*n.distance:
		#		k1 = tuple( map ( int , kp1[m.trainIdx].pt) )
		#		k2 = tuple( map ( int , kp2[m.trainIdx].pt) )
				#
		#		cv2.line(blank, k1, k2, (0,255,0), 1)
				#import ipdb; ipdb.set_trace()


		

		est = cv2.bitwise_and( gray, est)
		if fn == 100:
			cv2.imwrite('test1_30f.png', gray)
		elif fn == 120:
			cv2.imwrite('test2_60f.png', gray)
		while True:
			cv2.imshow('ray', blank)
			cv2.imshow('frame', gray)
			cv2.imshow('est', est)
			#cv2.imshow('scatter', test)
			key = cv2.waitKey(5) & 0xFF
			if key == ord('q'):
				return
			if key == ord('c'):
				break
		kp1 = kp2
		prev = gray
		blank[..., 0] = edges


if __name__ == '__main__':
	main()