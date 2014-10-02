import numpy as np
import cv2
import argparse
from template import template, maskTpl, visM
from template import Pts4 as HOOP

def cart2polar(matrix, center):
	# matrix should have 2 column 
	matrix = np.atleast_2d(matrix)
	m = matrix - center
	r = np.sqrt(np.sum(m**2,axis=1))
	theta = np.arctan2(m[:,0], m[:,1])
	return np.column_stack((r, theta))

hoops = HOOP / 2
hoops = hoops[[0,2]].astype(np.float32)

font = cv2.FONT_HERSHEY_PLAIN
drawing = False
moving = -1
movingP = None
movingRoi = None
resize = -1

def inverseM(m):
	src = np.array([[0,0],[500,500],[500,0],[0,500]],np.float32).reshape(-1,1,2)
	dst = cv2.perspectiveTransform(src, m)
	invM = cv2.getPerspectiveTransform(dst, src)
	#import pdb; pdb.set_trace()
	return invM

def movingSpot(roi):
	(x0,y0), (x1,y1), _ = roi
	return ((x0-10,y0+10), (x0+10, y0-10))

def resizeSpot(roi):
	(x0,y0), (x1,y1), _ = roi
	return ((x1-5,y1-5), (x1+5, y1+5))

def detectSpot(rois, x, y):
	for idx, ((x0,y0), (x1,y1)) in enumerate(rois):
		minx, maxx = min(x0,x1), max(x0,x1)
		miny, maxy = min(y0,y1), max(y0,y1)
		if (x <= maxx) and (x >= minx) and (y <= maxy) and (y >= miny):
			#print (x,y), (minx, miny), (maxx, maxy)
			return idx
	return -1

def capture(event, x, y, flags, param):
	rois = param
	global drawing, moving, movingP, movingRoi, resize
	movs = map(movingSpot, rois)
	ress = map(resizeSpot, rois)
	idxMov = detectSpot(movs, x, y)
	idxSize = detectSpot(ress, x, y)
	
	if event == cv2.EVENT_LBUTTONDBLCLK and idxMov > -1 :
		rois[idxMov][-1] = not rois[idxMov][-1]

	elif event == cv2.EVENT_LBUTTONDOWN and idxMov > -1 :
		moving = idxMov
		movingP = [x, y]
		movingRoi = rois[idxMov]

	elif event == cv2.EVENT_MOUSEMOVE and moving > -1:
		mvx = x - movingP[0]
		mvy = y - movingP[1] 
		(ori1x, ori1y), (ori2x, ori2y), flag = movingRoi
		rois[moving] = [(ori1x+mvx, ori1y+mvy), (ori2x+mvx, ori2y+mvy), flag]

	elif event == cv2.EVENT_LBUTTONUP and moving > -1:
		moving = -1

	elif event == cv2.EVENT_LBUTTONDOWN and idxSize > -1 :
		resize = idxSize
	elif event == cv2.EVENT_MOUSEMOVE and resize > -1:
		rois[resize][1] = (x, y)
	elif event == cv2.EVENT_LBUTTONUP and resize > -1:
		resize = -1

	elif event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		rois.append([(x, y), (x, y), True])

	elif event == cv2.EVENT_MOUSEMOVE and drawing:
		rois[-1][1] = (x, y)
		
	elif event == cv2.EVENT_LBUTTONUP and drawing:
		drawing = False
		(x0,y0), (x1,y1), _ = rois[-1]
		if abs(x1-x0) < 10  or (abs(y1-y0) < 10):
			del rois[-1]
		#rois[-1][1] = (x, y)
		

def globalMask(hsv):
	cmask1 = cv2.inRange(hsv, np.array((70., 0., 10.)), np.array((255.,255.,50.)))
	#return cmask1
	cmask2 = cv2.inRange(hsv, np.array((70., 0., 200.)), np.array((255.,255.,255.)))
	return cv2.bitwise_or(cmask1, cmask2)

def visHist(hist_item, nbins=32, height=48, width=384, color=(255,255,255)):
	bin_width = width/nbins
	hist = np.zeros((height, bin_width*nbins, 3), np.uint8)
	hist[..., :] = color
	bins = np.arange(nbins, dtype=np.int32).reshape(nbins,1)
	cv2.normalize(hist_item, hist_item, height, cv2.NORM_MINMAX)
	hist_item=np.int32(np.around(hist_item))
	pts = np.column_stack((bins,hist_item))
	for x,y in enumerate(hist_item):
		cv2.rectangle(hist,(x*bin_width,y),
			(x*bin_width + bin_width-1,height),(0,0,0),-1)
	#hist=np.flipud(hist)

	return np.swapaxes(hist, 0, 1)

def rois2court(rois, invM):
	if len(rois) == 0 : return []
	pts = []
	flags = []
	for (x0,y0), (x1,y1), flag in rois:
		if abs(y1-y0)>10 and abs(x1-x0)>10:
			pts.append( (x0+(x1-x0)/2,y1+40) )
			flags.append(flag)
	if len(pts)>0:
		pts = np.float32(pts).reshape(-1, 1, 2)
		ptsCourt = cv2.perspectiveTransform(pts, invM).reshape(-1,2)
		#import pdb; pdb.set_trace()
		return zip( np.int32(ptsCourt), flags)
	return []
		
def renderOnCourt(court, rois):
	for (x, y), flag in rois:
		if flag : 
			color = (57, 184, 138)
		else : 
			color = (46, 0, 184)
		cv2.circle(court, (x,y), 10, color, -1)

def highlight(img, rois, hoop):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	m,n = hsv.shape[:2]
	blank = np.zeros_like(img)
	cv2.ellipse(blank, tuple(hoop), (20, 10), 0, 0, 360, (255,255,255), 2)
	for idx, ((x0,y0), (x1,y1), flag) in enumerate(rois):
		if abs(y1-y0)>10 and abs(x1-x0)>10:
			if flag : 
				color = (57, 184, 138)
				category = '+'
			else : 
				color = (46, 0, 184)
				category = '-'
			mask = globalMask(hsv)
			cv2.rectangle(mask, (x0,y0), (x1,y1), 255, -1)
			hist_item = cv2.calcHist( [hsv], [0], mask, [32], [0, 180] )
			cv2.rectangle(blank, (x0,y0), (x1,y1), color, 2)
			cv2.rectangle(blank, (x0,y0-15), (x1,y0), color, -1)
			pos = (x0+(x1-x0)/2,y1+40)
			cv2.ellipse(blank, pos, (20, 10), 0, 0, 360, color, 2)
			if flag:
				degree = np.arctan2(pos[1]-hoop[1], pos[0]-hoop[0]) * 180 / np.pi
				#cv2.ellipse(blank, pos, (100, 80), degree, 100, 240, color, 4)
			cv2.line(blank, (x0+(x1-x0)/2,y1) ,pos, color, 1)
			cv2.line(blank, pos, tuple(hoop), color, 2)
			cv2.circle(blank,(x0+(x1-x0)/2,y1+40), 5, color, -1 )
			cv2.putText(blank, "%s %s"%(idx, category), (x0,y0-2), font, 1, (255,255,255), 1)
			#histFeature = visHist(hist_item, height=32, width=abs(y1-y0), color=color)
			#mh,nh,_ = histFeature.shape
			#if x1+nh+1 < n:
			#	blank[y0:y0+mh, x1+1:x1+nh+1] = histFeature
			#else:
			#	histFeature=np.fliplr(histFeature)
			#	blank[y0:y0+mh, x0-nh-1:x0-1] = histFeature
			#polar = cart2polar((x0+(x1-x0)/2,y1+40), hoop)
			#cv2.putText(blank, "%.0f %.2f"%(polar[0][0], polar[0][1]), (x0+(x1-x0)/2,y1+40), font, 1, (255,255,255), 1)
	
	bgGray = cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)
	ret, bgMask = cv2.threshold(bgGray, 10, 255, cv2.THRESH_BINARY)
	bgMask_inv = cv2.bitwise_not(bgMask)
	img_bg = cv2.bitwise_and(img, img, mask=bgMask_inv)
	img_fg = cv2.bitwise_and(blank, blank, mask=bgMask)
	cv2.add(img_bg, img_fg, dst=img)

def buildFeature(img, rois):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	m, n = img.shape[:2]
	mask1 = np.zeros((m,n), np.uint8)
	mask2 = np.zeros((m,n), np.uint8)
	gMask = globalMask(hsv)
	bMask = cv2.inRange(hsv, np.array((70., 0., 10.)), np.array((255.,255.,50.)))
	wMask = cv2.inRange(hsv, np.array((70., 0., 200.)), np.array((255.,255.,255.)))
	for (x0,y0), (x1,y1), flag in rois:
		if flag:
			cv2.rectangle(mask1, (x0,y0), (x1,y1), 255, -1)
		else:
			cv2.rectangle(mask2, (x0,y0), (x1,y1), 255, -1)
	cv2.bitwise_and(wMask, mask1, dst=mask1)
	cv2.bitwise_and(bMask, mask2, dst=mask2)
	f1 = cv2.calcHist([hsv],[0,2],mask1,[16, 4],[70, 180, 10, 255])
	f2 = cv2.calcHist([hsv],[0,2],mask2,[16, 4],[70, 180, 10, 255])
	cv2.normalize(f1,f1,0,255,cv2.NORM_MINMAX)
	cv2.normalize(f2,f2,0,255,cv2.NORM_MINMAX)
	return f1, f2

def meanShift(hsv, f1, f2, rois, mask):
	if f1 is not None:
		term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
		#import pdb; pdb.set_trace()
		ff1 = cv2.normalize(f1,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)
		ff2 = cv2.normalize(f2,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)
		dst1 = cv2.calcBackProject([hsv], [0,2], ff1, [70, 180, 10, 255], 1)
		dst2 = cv2.calcBackProject([hsv], [0,2], ff2, [70, 180, 10, 255], 1)
		bMask = cv2.inRange(hsv, np.array((70., 0., 10.)), np.array((255.,255.,50.)))
		wMask = cv2.inRange(hsv, np.array((70., 0., 200.)), np.array((255.,255.,255.)))
		dst1 = cv2.bitwise_and(dst1, mask)
		dst2 = cv2.bitwise_and(dst2, mask)
		dst1 = cv2.bitwise_and(dst1, wMask)
		dst2 = cv2.bitwise_and(dst2, bMask)
		dstVis1 = cv2.applyColorMap(dst1, cv2.COLORMAP_JET)
		dstVis2 = cv2.applyColorMap(dst2, cv2.COLORMAP_JET)

		nRois = []
		for (x0,y0), (x1,y1), flag in rois:
			if flag:
				x,y,w,h = cv2.meanShift(dst1, (x0,y0,x1-x0,y1-y0), term_crit)[1]
				nRois.append( [(x, y), (x+w, y+h), flag] )
				dst1[y:y+h, x:x+w] = 0
			else:
				x,y,w,h = cv2.meanShift(dst2, (x0,y0,x1-x0,y1-y0), term_crit)[1]
				nRois.append( [(x, y), (x+w, y+h), flag] )
				dst2[y:y+h, x:x+w] = 0
		
		#cv2.imshow('dst1', dstVis1)
		#cv2.imshow('dst2', dstVis2)
		return nRois
	return []

def trajectory(args):
	src, ms, dst, rou = args['src'], args['tran'], args['dst'], args['route']
	debugFlag = args['noDebug']
	Ms = np.load(ms)
	if debugFlag is not None : 
		debugFlag = False
	else :
		debugFlag = True
	cap = cv2.VideoCapture(src)
	num = 0

	cv2.namedWindow('frame')
	frame_w, frame_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	frame_w, frame_h = int(frame_w), int(frame_h)
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(dst, fourcc, 20.0, (frame_w, frame_h))

	tplMask = maskTpl()[...,0]
	tpl = template()
	tplC = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
	tplM, tplN = tplMask.shape
	routes = []
	rois = []
	f1, f2 = np.zeros((16,4)), np.zeros((16,4))
	dst1 = np.zeros((frame_h, frame_w), np.uint8)
	dst2 = np.zeros((frame_h, frame_w), np.uint8)
	
	while(cap.isOpened()):
		ret, frame = cap.read()
		if not ret : break
		courtMask = cv2.warpPerspective(tplMask, Ms[num], (frame_w, frame_h),
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
		court = cv2.warpPerspective(tplC, Ms[num], (frame_w, frame_h))
		hoop = cv2.perspectiveTransform(hoops.reshape(-1,1,2), Ms[num]).reshape(-1,2)
		if hoop[0,0] < 0 : hoop = hoop[1]
		else: hoop = hoop[0]
		invM = inverseM(Ms[num])
		courtMask[550:680, 240:955] = 0
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		courtMask = cv2.bitwise_and( globalMask(hsv), courtMask)
		rois = meanShift(hsv, f1, f2, rois, courtMask)
		cv2.setMouseCallback('frame', capture, param=(rois))
		
		#maskFrame = cv2.bitwise_and(frame, frame, mask=courtMask)

		while num==0 or debugFlag:
			fcopy = frame.copy()
			tplCopy = tplC.copy()
			highlight(fcopy, rois, hoop)
			roisCourt = rois2court(rois, invM)
			renderOnCourt(tplCopy, roisCourt)
			tplCopy = cv2.resize(tplCopy, (tplN/5, tplM/5))
			cv2.putText(fcopy, "#%s"%(num), (50,50), font, 1, (255,255,255), 1)
			fcopy[-tplM/5:, -tplN/5:] = tplCopy

			cv2.imshow('frame', fcopy)
			#cv2.imshow('court', court)
			k = cv2.waitKey(150) & 0xFF 
			if k == ord('c'):
				f1, f2 = buildFeature(frame, rois)
				if len(roisCourt) > 0:
					routes.append(roisCourt)
				out.write(fcopy)
				break
			elif k == ord('q'):
				cap.release()
				break
		if not debugFlag:
			fcopy = frame.copy()
			tplCopy = tplC.copy()
			highlight(fcopy, rois)
			roisCourt = rois2court(rois, invM)
			renderOnCourt(tplCopy, roisCourt)
			tplCopy = cv2.resize(tplCopy, (tplN/5, tplM/5))
			cv2.putText(fcopy, "#%s"%(num), (50,50), font, 1, (255,255,255), 1)
			fcopy[-tplM/5:, -tplN/5:] = tplCopy
			f1, f2 = buildFeature(frame, rois)
			if len(roisCourt) > 0:
				routes.append(roisCourt)
			out.write(fcopy)
			print num
		num += 1

	np.save(rou, np.array(routes))
	out.release()
	cap.release()
	cv2.destroyAllWindows()



def main():
	parser = argparse.ArgumentParser(description="detect trajectory")
	parser.add_argument('-s', '--src', help='case folder name', required=True)

	args = vars(parser.parse_args())
	if args['src'][-1] != '/':
		args['src'] = args['src'] + "/"
	params = {'src' : "%sgrid.avi"%args['src'], 
			"tran": "%sgrid.m.npy"%args['src'],
			"dst" : "%sdetect.avi"%args['src'],
			"route": "%sroute.npy"%args['src'],
			"noDebug" : None}
	trajectory(params)

def tracking_wrapper(src):
	args = {'src':src}
	if args['src'][-1] != '/':
		args['src'] = args['src'] + "/"
	params = {'src' : "%sgrid.avi"%args['src'], 
			"tran": "%sgrid.m.npy"%args['src'],
			"dst" : "%sdetect.avi"%args['src'],
			"route": "%sroute.npy"%args['src'],
			"noDebug" : None}
	trajectory(params)

def demo():
	args = {'src' : "../data/heat3.avi", 
			"tran": "../data/heat3.1.grid.m.npy",
			"dst" : "../data/heat3.1.detect.avi",
			"route": "../data/heat3.1.route.npy",
			"noDebug" : None}

	trajectory(args)

if __name__ == '__main__':
	#demo()
	main()