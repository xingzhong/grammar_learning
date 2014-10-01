import numpy as np
import cv2

def remove_banner(frame):
	frame[550:680, 240:955] = 0
	return frame

def denoise(frame):
	frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	ret,threshColor = cv2.threshold(frameHSV[:,:,0], 30, 179, cv2.THRESH_BINARY_INV)
	threshColorMask = cv2.bitwise_and(frame, frame, mask = threshColor)
	blank = np.zeros_like(threshColor)
	gray =  cv2.cvtColor(threshColorMask, cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(gray,100,255,0)
	image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	blank = cv2.drawContours(blank, contours, -1, 255, 1)
	kernel = np.ones((3,3),np.uint8)
	dilate = cv2.dilate(blank,kernel,iterations = 1)
	closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
	closing = remove_banner(closing)
	return closing

if __name__ == '__main__':
	from template import template, POI
	from sklearn.neighbors import NearestNeighbors
	font = cv2.FONT_HERSHEY_SIMPLEX
	name = "../data/heat1"
	cap = cv2.VideoCapture("%s.avi"%name)
	cv2.namedWindow('Mask')
	num = 0
	courtPts = [(299, 269), (10, 269), (10, 513), (299, 513)]
	framePts = [(671, 330), (277, 305), (126, 411), (579, 441)]
	M = cv2.getPerspectiveTransform(np.float32(courtPts), np.float32(framePts))
	lastPts = []
	while(cap.isOpened()):
		ret, frame = cap.read()
		if not ret: break
		cnt = True
		tpl = template()
		frame_h, frame_w, _ = frame.shape
		closing = denoise(frame)
		nbKeys = []
		if len(lastPts):
			X = np.array(lastPts)
			closingGray = closing[:,:,0]+  closing[:,:,1]+  closing[:,:,2]
			closeIdx = np.vstack(np.nonzero(closingGray)[::-1]).T
			nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(closeIdx)
			distances, indices = nbrs.kneighbors(X)
			nbKeys = closeIdx[indices].reshape((-1,2))
			dM = cv2.estimateRigidTransform(np.float32(X), np.float32(nbKeys), False)
			dM = np.vstack( (dM , np.array([0,0,1])))
			print distances.shape
			M = np.dot(dM, M)
			#import pdb; pdb.set_trace()
		cRot = cv2.warpPerspective(tpl, M, (frame_w, frame_h))
		dns = cv2.bitwise_and(cRot, closing)
		cv2.addWeighted(cRot, 0.3, frame, 0.7, 0, dst=frame)
		for p1, p2 in zip(lastPts, nbKeys):
			cv2.circle(closing, p1, 4, (255, 0, 0), 2)
			cv2.circle(closing, tuple(p2), 6, (0,255,0), 2)
			cv2.line(closing, p1, tuple(p2), (255,255,0), 2)

		POIs = cv2.perspectiveTransform(np.float32(POI).reshape(-1,1,2), M).reshape((-1,2)).astype(int)
		idx = (POIs[:,1]<720) * (POIs[:,0]<1280) * (POIs[:,1]>0) * (POIs[:,0]>0)
		lastPts = []
		for x, y in POIs[idx]:
			if np.sum(dns[y, x]) > 0:
				cv2.circle(dns, (x,y), 6, (0,0,255), 1)
				lastPts.append((x,y))
			else:
				cv2.circle(dns, (x,y), 6, (0,0,32), 1)
		while cnt:
			cv2.imshow('frame', frame)
			cv2.imshow('closing', closing)
			cv2.imshow('dns', dns)
			k = cv2.waitKey(100) & 0xFF 
			if k == ord('c'):
				break
			elif k == ord('q'):
				cap.release()
				break
		num += 1