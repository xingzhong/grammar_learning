from template import template, visM
import cv2
import numpy as np

def nothing(x): pass


def initM():
	courtPts = [[613, 161], [613, 770], [394, 460], [835, 460]]
	framePts = [[619, 260], [626, 674], [58, 437], [1175, 424]]
	M = cv2.getPerspectiveTransform(np.float32(courtPts), np.float32(framePts))
	return M

def main():
	cv2.namedWindow('image')
	cv2.createTrackbar('X','image',0,4000,nothing)
	cv2.createTrackbar('Y','image',0,4000,nothing)
	cv2.createTrackbar('F','image',0,200,nothing)

	cap = cv2.VideoCapture('sep/0_0.avi')
	fn = 0
	ret, iframe = cap.read()
	H, W, _ = iframe.shape
	tpl = template()
	m = initM()

	print m
	while(cap.isOpened()):
		ret, frame = cap.read()
		if not ret: break
		fn += 1
		while True:
			img = frame.copy()
			x = cv2.getTrackbarPos('X','image')
			y = cv2.getTrackbarPos('Y','image')
			f = cv2.getTrackbarPos('F','image')

			x -= 2000
			y -= 2000
			f = f/100.0

			translate = np.array([[1,0,x], [0,1,y], [0,0,1]])
			scale = np.array([[f,0,0], [0,f,0], [0,0,1]])
			
			mm = np.dot(translate, scale)

			visTpl = cv2.warpPerspective(tpl, np.dot(mm, m), (1280, 720))
			mask_inv = cv2.bitwise_not(visTpl)
			img[..., 1] = visTpl
			cv2.imshow('image', img)
			key = cv2.waitKey(5) & 0xFF
			if key == ord('q'):
				return
			if key == ord('c'):
				break

if __name__ == '__main__':
	main()