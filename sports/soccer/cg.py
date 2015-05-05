import cv2
import csv
import pandas as pd
import numpy as np
from sklearn.mixture import GMM, VBGMM, DPGMM
from collections import Counter

PWD = '/home/xingzhong/Documents/sports_data/home.ifi.uio.no/paalh/dataset/alfheim/2013-11-03/zxy/'
HEADERS = ['timestamp','tag_id','x_pos','y_pos','heading','direction','energy','speed','total_distance']
font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
R = 10
COLORS = [(58,43,217), (5,203,242), (53,104,242)]

# (20, 444), (681, 22)
def maxXY(x, y):
	b1 = (681-20)/105.0
	b2 = (22-444)/68.0
	return int(b1*x+20), int(b2*y+444)

def court():
	bg = cv2.imread('court.jpg', cv2.IMREAD_COLOR)
	return bg

def MapMetric(data, pic, model):
	if len(data) <= 3 :
		return model
	xv, yv = int(.4*data.x_pos.var()), int(.4*data.y_pos.var())
	xm, ym = maxXY(data.x_pos.mean(), data.y_pos.mean())
	dm = data.direction.mean()
	dx, dy = int(xm+xv*np.sin(dm)), int(ym-yv*np.cos(dm))
	#cv2.ellipse(pic, (xm, ym), (xv, yv), 0, 0, 360, (255,255,0), 2)
	
	fModel = GMM(n_components=3, covariance_type='diag', min_covar=20.0,
			n_init=30, n_iter=10, params='mw', init_params='w')
	#fModel = DPGMM(n_components=5, covariance_type='diag', 
	#	alpha=0.3, params='cmw', init_params='cw')
	

	X = data[['x_pos', 'y_pos']].values
	if model is None:
		fModel.means_ = np.array([[52, 34], [25,34], [75,34]])
		#fModel.weights_ = np.array([0.4, 0.4, 0.2])
	else:
		fModel.means_ = model.means_.copy() + 1*np.random.randn(3, 2)
		#fModel.weights_ =  model.weights_.copy()

	fModel.covars_ = np.array([[25,200], [25,200], [25,100] ])
	fModel.fit(X)
	orderID =  np.argsort(fModel.means_[:,0])
	fModel.means_ = fModel.means_[orderID]
	fModel.weights_ = fModel.weights_[orderID]
	fModel.covars_ = fModel.covars_[orderID]
	
	#cbar = map(lambda x: COLORS[x], np.argsort(fModel.means_[:,0]))
	Y = fModel.predict(X)
	f = map(str, Counter(Y).values())
	cv2.putText(pic, "%s"%"-".join(f), (450,15), font, 1, (255,255,255), 1)
	c = map(lambda x: COLORS[x], Y)
	#import ipdb; ipdb.set_trace()
	#cv2.circle(pic, (xm, ym), 4, c, -1)
	#cv2.line(pic,(xm, ym), (dx, dy), c, 1)
	#import ipdb; ipdb.set_trace()
	print fModel.means_
	print fModel.covars_
	#print fModel.weights_
	for idx, pid in enumerate(data.index):
		dd = data.loc[pid]
		x, y = dd.x_pos, dd.y_pos
		speed = int(196 + (255-196)*dd.speed/14)
		xx, yy = maxXY(x, y)
		dx, dy = int(xx+R*np.sin(dd.heading)), int(yy-R*np.cos(dd.heading))
		dx2, dy2 = int(xx-R*np.sin(dd.direction)), int(yy+R*np.cos(dd.direction))
		dx3, dy3 = int(xx-2*R*np.sin(dd.direction)), int(yy+2*R*np.cos(dd.direction))
		dx4, dy4 = int(xx-3*R*np.sin(dd.direction)), int(yy+3*R*np.cos(dd.direction))
		cv2.circle(pic,(xx, yy), R, COLORS[Y[idx]], 2)
		cv2.line(pic,(xx, yy), (dx, dy), COLORS[Y[idx]], 2)
		cv2.line(pic,(dx2, dy2), (dx3, dy3), COLORS[Y[idx]], 6)
		cv2.line(pic,(dx3, dy3), (dx4, dy4), COLORS[Y[idx]], 1)
		cv2.putText(pic, "%s"%pid, (dx, dy), cv2.FONT_HERSHEY_PLAIN , 1, (0,255,255), 1)

	for n, color in enumerate(COLORS):
		v, w = np.linalg.eigh(fModel._get_covars()[n][:2, :2])
		u = w[0] / np.linalg.norm(w[0])
		angle = np.arctan2(u[1], u[0])
		angle = 180 * angle / np.pi  # convert to degrees
		#v /= 9
		x, y = fModel.means_[n, :2]
		x, y = maxXY(x, y)
		vx, vy = int(v[0]), int(v[1])
		#import ipdb; ipdb.set_trace()
		cv2.ellipse(pic, (x, y), (vx, vy), 0, 0, 360, color, 2)
		cv2.circle(pic, (x, y), 5, color, -1)
	#import ipdb; ipdb.set_trace()
	return fModel
	#

def mapInfo(d, pic):
	for pid in d.index:
		dd = d.loc[pid]
		x, y = dd.x_pos, dd.y_pos
		speed = int(196 + (255-196)*dd.speed/14)
		xx, yy = maxXY(x, y)
		dx, dy = int(xx+R*np.sin(dd.heading)), int(yy-R*np.cos(dd.heading))
		dx2, dy2 = int(xx-R*np.sin(dd.direction)), int(yy+R*np.cos(dd.direction))
		dx3, dy3 = int(xx-2*R*np.sin(dd.direction)), int(yy+2*R*np.cos(dd.direction))
		dx4, dy4 = int(xx-3*R*np.sin(dd.direction)), int(yy+3*R*np.cos(dd.direction))
		cv2.circle(pic,(xx, yy), R, (0,0,speed), 2)
		cv2.line(pic,(xx, yy), (dx, dy), (0,0,speed), 2)
		cv2.line(pic,(dx2, dy2), (dx3, dy3), (0,0,speed), 6)
		cv2.line(pic,(dx3, dy3), (dx4, dy4), (0,0,speed), 1)
		cv2.putText(pic, "%s"%pid, (dx, dy), cv2.FONT_HERSHEY_PLAIN , 1, (0,255,255), 1)
	#return dic

def main(name):
	bound = 5
	data = pd.read_csv(PWD+name, names=HEADERS, index_col=[0,1])
	data = data[(data.x_pos < 105 - bound) & (data.x_pos > 0 + bound) 
			& (data.y_pos>0 + bound) & (data.y_pos<68-bound)]
	#import ipdb; ipdb.set_trace()
	#out = cv2.VideoWriter('%s.avi'%name,fourcc, 20.0, (700,462))
	pIndex = data.index.get_level_values(0)
	bg = court()
	pic = bg.copy()
	flag = True
	model = None
	for time in pIndex[::10]:
		cv2.copyMakeBorder(bg,0,0,0,0,cv2.BORDER_REPLICATE, dst=pic)
		d = data.loc[time]
		print time
		cv2.putText(pic, "%s"%time, (15,15), font, 1, (255,255,255), 1)
		#mapInfo(d, pic)
		model = MapMetric(d, pic, model)
		#out.write(pic)
		cv2.imshow('frame', pic)

		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
	
	out.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main('2013-11-03_tromso_stromsgodset_first.csv')

