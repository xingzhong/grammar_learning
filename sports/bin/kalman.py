import cv2
from pykalman import KalmanFilter
import numpy as np
from collections import deque

def demo():
	img = np.zeros((768,1024,3), np.uint8)
	trajs = deque(maxlen=200)
	trajs_filterd = deque(maxlen=200)

	def draw(event,x,y,flags,param):
		kf, fm, fo = param
		if event == cv2.EVENT_MOUSEMOVE:
			fm[:], fo[:] = kf.filter_update(fm, fo, np.array([x, y] ) )
			fx = int(fm[0])
			fy = int(fm[1])
			trajs.append((x,y))
			trajs_filterd.append( (fx, fy) )
			
	cv2.namedWindow('image')
	

	kf = KalmanFilter(n_dim_state=4, n_dim_obs=2)
	kf.transition_matrices = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]])
	kf.observation_matrices = np.array([[1,0,0,0], [0,1,0,0]])
	kf.transition_covariance = 1e-3 * np.eye(4)
	kf.observation_covariance = 10 * np.eye(2)
	kf.initial_state_mean = np.array([0,0,0,0])
	kf.initial_state_covariance = np.eye(4)
	fm = kf.initial_state_mean
	fo = kf.initial_state_covariance

	cv2.setMouseCallback('image',draw,(kf, fm, fo))
	
	while True:
		img = np.zeros((768,1024,3), np.uint8)
		for i in range(len(trajs) - 1):
			cv2.line(img, trajs[i], trajs[i+1], (0,255,0), 1 )
			cv2.line(img, trajs_filterd[i], trajs_filterd[i+1], (0,0,255), 1 )
		map(lambda pt: cv2.circle(img,pt,3,(0,255,0), 1), trajs)
		map(lambda pt: cv2.circle(img,pt,3,(0,0,255), 1), trajs_filterd)
		cv2.imshow('image',img)
		if cv2.waitKey(5) & 0xFF == ord('q'):
			return 

if __name__ == '__main__':
	demo()