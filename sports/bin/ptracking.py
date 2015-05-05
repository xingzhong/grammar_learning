import numpy as np
import cv2


FONT = cv2.FONT_HERSHEY_SIMPLEX
kernel = np.ones((5,5),np.uint8)


def renderLines(img, lines, cnt, dx=0, dy=0):
	for idx, line in enumerate(lines):
		if idx >= cnt : break
		rho, theta = line[0]
		print theta
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 2000*(-b))
		y1 = int(y0 + 2000*(a))
		x2 = int(x0 - 2000*(-b))
		y2 = int(y0 - 2000*(a))
		cv2.line(img,(dx+x1,dy+y1),(dx+x2,dy+y2),255,2)
		

def main():
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
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		ret,threshColor = cv2.threshold(frameHSV[:,:,0], 30, 180, cv2.THRESH_BINARY_INV)
		ret,threshInner = cv2.threshold(frameHSV[:,:,1], 180, 255, cv2.THRESH_BINARY)
		threshColor = cv2.inRange(frameHSV, np.array([11,47,151]), np.array([16, 255, 255]))


		edges = cv2.Canny(threshColor,50,150,apertureSize = 3)
		dedges = cv2.dilate(edges, np.ones((2, 2),np.uint8))
		dedges[550:680, 240:955] = 0

		line1 = cv2.HoughLines(dedges[H/6:3*H/4, :], 1, np.pi/180, 250 )
		line2 = cv2.HoughLines(dedges[:, :W/2], 1, np.pi/180, 150 )
		line3 = cv2.HoughLines(dedges[:, W/2:], 1, np.pi/180, 150 )
		

		if line1 is None : line1 = []
		if line2 is None : line2 = []
		if line3 is None : line3 = []
		line1 = filter( lambda x: abs(x[0][1] - np.pi/2)< 0.1 , line1)
		line2 = filter( lambda x: abs(x[0][1] - np.pi/3.5)< 0.1 , line2)
		line3 = filter( lambda x: abs(x[0][1] - np.pi/1.5)< 0.1, line3)

		blank = np.zeros_like(edges)
		#renderLines(edges, line1, 5, dy = H/6)
		#renderLines(edges, line2, 1)	
		#renderLines(edges, line3, 1, dx = W/2)

			#cv2.line(blank,(x1,y1),(x2,y2),255,2)
		#laplacian = cv2.resize(laplacian,(16*30, 9*30), interpolation = cv2.INTER_AREA)
		#sobely = cv2.Sobel(threshColor,cv2.CV_64F,0,1,ksize=5)
		#sobelx = cv2.Sobel(threshColor,cv2.CV_64F,1,0,ksize=5)
		cv2.putText(frame, "#f %d"%fn ,(10, 30), FONT, 1,(255,255,255),1,cv2.LINE_AA)
		#cv2.imshow('sobely',sobely)
		
		#cv2.imshow('edges',edges)
		cv2.imshow('frame',frameHSV)
		#cv2.imshow('threshColor',threshColor)
		#cv2.imshow('laplacian',opening[H/6:2*H/3, :])
		#frame = cv2.bitwise_and(frame, frame, mask=mask)
		cv2.imshow('frame2',dedges)
		#cv2.imshow('mask',mask)
		#cv2.imshow('frame3',frameHSV[..., 2])
		if cv2.waitKey(5) & 0xFF == ord('q'):
			break

if __name__ == '__main__':
	main()