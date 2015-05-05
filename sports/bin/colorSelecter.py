import numpy as np
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
def nothing(x):
	pass


# for floor best params (11,47,151), (16, 255, 255)
# for ball (0,81,60), (10, 150, 110)
# for heat (120,11,130), (153, 71, 255)
# for spurs (10,0,0), (180,255,057)
def main():
	cv2.namedWindow('frame')
	cv2.createTrackbar('H1','frame',0,179,nothing)
	cv2.createTrackbar('S1','frame',0,254,nothing)
	cv2.createTrackbar('V1','frame',0,254,nothing)
	cv2.createTrackbar('H2','frame',1,180,nothing)
	cv2.createTrackbar('S2','frame',1,255,nothing)
	cv2.createTrackbar('V2','frame',1,255,nothing)
	cap = cv2.VideoCapture('sep/0_0.avi')
	#cap = cv2.VideoCapture('sep/1_12557.avi')
	#cap = cv2.VideoCapture('sep/2_14117.avi')
	fn = 0
	ret, iframe = cap.read()
	H, W, _ = iframe.shape

	
	while(cap.isOpened()):
		ret, frame = cap.read()
		if not ret: break
		fn += 1
		frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		while True:
			h1 = cv2.getTrackbarPos('H1','frame')
			s1 = cv2.getTrackbarPos('S1','frame')
			v1 = cv2.getTrackbarPos('V1','frame')
			h2 = cv2.getTrackbarPos('H2','frame')
			s2 = cv2.getTrackbarPos('S2','frame')
			v2 = cv2.getTrackbarPos('V2','frame')
			mask = cv2.inRange(frameHSV, np.array([h1,s1,v1]), np.array([h2,s2,v2]))
			filterd = cv2.bitwise_and(frame, frame, mask=mask)
			cv2.imshow('frame',mask)
			cv2.imshow('raw',frame)
			key = cv2.waitKey(5) & 0xFF
			if key == ord('q'):
				return
			if key == ord('c'):
				break

if __name__ == '__main__':
	main()