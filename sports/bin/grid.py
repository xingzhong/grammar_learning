## do court mapping 

import numpy as np
import cv2
from template import template, POI
from denoise import denoise
from collections import deque
import itertools
from sklearn.neighbors import NearestNeighbors
import argparse
import csv
import os

COLORS = [(255,0,0), (0,255,0), (0,255,255), (255,255,0)]
font = cv2.FONT_HERSHEY_SIMPLEX
drawing = -1


def clickCourt(event, x, y, flags, param):
	img, num, courtPts, rawPts = param
	if event == cv2.EVENT_LBUTTONDBLCLK:
		distances = np.linalg.norm(np.array([x,y]) - POI, axis=1)
		if min(distances) < 15 :
			x, y = POI[np.argmin(distances)]
		courtPts.append((x, y))
		rawPts.append((10, 10))

def clickRaw(event, x, y, flags, param):
	global drawing
	img, num, courtPts, rawPts, players = param
	if event == cv2.EVENT_LBUTTONDOWN:
		if len(players) > 0 :
			tgt = np.vstack((np.array(rawPts), np.array(players)))
		else:
			tgt = np.array(rawPts)
		distances = np.linalg.norm(np.array([x,y]) - tgt, axis=1)
		if min(distances) < 15 :
			drawing = np.argmin(distances)
			if drawing < len(rawPts):
				rawPts[drawing] = (x, y)
			else:
				players[drawing-len(rawPts)] = (x, y)
			
	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing > -1:
			if drawing < len(rawPts):
				rawPts[drawing] = (x, y)
			else:
				players[drawing-len(rawPts)] = (x, y)

	elif event == cv2.EVENT_LBUTTONUP:
		drawing = -1

	elif event == cv2.EVENT_LBUTTONDBLCLK:
		players.append((x, y))

def pts_span(pt):
	x, y = pt
	return [ (x-1, y), (x, y-1), (x, y), (x, y+1), (x+1, y)]

		
def sub_mapping(frame, tpl, num, courtPts=[], rawPts=[], players=[], matchedKey=[]) :
	frame_h, frame_w, _ = frame.shape
	edgeRaw = cv2.Canny(frame,100,200)
	oldM = np.zeros((3,3), dtype=np.float32)
	cRot = np.zeros_like(frame)
	cPlayers = []
	score = 0
	cv2.setMouseCallback('court', clickCourt, param=(tpl, num, courtPts, rawPts))
	cv2.setMouseCallback('dst', clickRaw, param=(frame, num, courtPts, rawPts, players))
	court, raw = tpl.copy(), frame.copy()
	dns = denoise(frame)
	dns = cv2.cvtColor(dns, cv2.COLOR_GRAY2BGR)
	nbKeys = []
	if len(matchedKey)>0:
		X = np.array(matchedKey)
		dnsGray = dns[:,:,0]+  dns[:,:,1]+  dns[:,:,2]
		dnsIdx = np.vstack(np.nonzero(dnsGray)[::-1]).T
		nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(dnsIdx)
		distances, indices = nbrs.kneighbors(X)
		nbKeys = dnsIdx[indices].reshape((-1,2))
		dM = cv2.estimateRigidTransform(np.float32(X), np.float32(nbKeys), False)
		dM = np.vstack( (dM , np.array([0,0,1])))
		cv2.warpPerspective(tpl, dM, (frame_w, frame_h), dst=cRot)
		
	dnsCopy = dns.copy()
	newKeys = []
	while(1):
		
		cv2.copyMakeBorder(tpl, 0,0,0,0,0, dst=court)
		cv2.copyMakeBorder(frame, 0,0,0,0,0, dst=raw)
		for (x, y) in POI:
			cv2.circle(court, (x, y), 2, (0,0,255), -1)
		#for (x, y) in HOOPS:
		#	cv2.circle(court, (x, y), 2, (0,0,255), -1)
		for idx, (x,y) in enumerate(courtPts):
			cv2.circle(court, (x, y), 5, COLORS[idx], 2)

		for idx, player in enumerate(players):
			cv2.ellipse(raw, player, (25, 15), 0, 0, 360, COLORS[idx], 2)
			cv2.circle(raw, player, 3, COLORS[idx], -1)
			cv2.line(raw, player, (player[0]+25, player[1]), COLORS[idx], 2)
		for x, y in matchedKey:
			cv2.circle(dnsCopy, (x,y), 4, (0,255,0), 1)
		for x, y in nbKeys:
			cv2.circle(dnsCopy, (x,y), 6, (255,0,0), 1)

		if len(courtPts) == 4 :
			M = cv2.getPerspectiveTransform(np.float32(courtPts), np.float32(rawPts))
			N = cv2.getPerspectiveTransform(np.float32(rawPts), np.float32(courtPts))
			if np.linalg.norm(M-oldM)>0.01:
				print "call warpPerspective"
				cv2.warpPerspective(tpl, M, (frame_w, frame_h), dst=cRot)
				POIs = cv2.perspectiveTransform(np.float32(POI).reshape(-1,1,2), M).reshape((-1,2)).astype(int)
				cv2.bitwise_and(cRot, dns, dst=dnsCopy)
				idx = (POIs[:,1]<720) * (POIs[:,0]<1280) * (POIs[:,1]>0) * (POIs[:,0]>0)
				newKeys = []
				for x, y in POIs[idx]:
					#import pdb; pdb.set_trace()
					if np.sum(dnsCopy[y, x]) > 0:
						cv2.circle(dnsCopy, (x,y), 4, (0,0,255), 2)
						newKeys.append((x,y))
					else:
						cv2.circle(dnsCopy, (x,y), 4, (0,0,32), 2)
				score = np.sum(dnsCopy)/float(np.sum(cRot))
				oldM = M
			cv2.addWeighted(cRot, 0.3, dns, 0.7, 0, dst=raw)
			
			if len(players)>0:
				cPlayers = cv2.perspectiveTransform(np.float32(players).reshape(-1,1,2), N)
				cPlayers = cPlayers.reshape(-1,2).astype(int)
		
		for idx, (x,y) in enumerate(rawPts):
			cv2.circle(raw, (x, y), 6, COLORS[idx], 3)

		for idx, player in enumerate(cPlayers):
			cv2.circle(court, tuple(player), 5, COLORS[idx], -1)

		cv2.putText(raw, "# %s"%num, (30,30), font, 1, (255,255,255), 1)
		cv2.putText(raw, "S %.2f %%"%(100*score), (30,70), font, 1, (255,255,255), 1)
		cv2.putText(dnsCopy, "# %s"%num, (30,30), font, 1, (255,255,255), 1)
		cv2.putText(dnsCopy, "S %.2f %%"%(100*score), (30,70), font, 1, (255,255,255), 1)
		cv2.imshow('dst', raw)
		cv2.imshow('court', court)

		k = cv2.waitKey(20) & 0xFF 
		if k == ord('c'):
			print courtPts, rawPts
			return courtPts, rawPts
		elif k == ord('q'):
			return 

def mapping(args):
	cv2.namedWindow('dst')
	cv2.namedWindow('court')
	src, dst = args['src'], args['dst']
	cap = cv2.VideoCapture(src)
	num = 1
	Ms, courtPts, rawPts, players, playersInCourt = [], [], [], [], []
	matchedKey=[]
	while(cap.isOpened()):
		ret, frame = cap.read()
		if not ret: break
		tpl = template()
		tpl = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
		courtPts, rawPts = deque(courtPts, maxlen=4), deque(rawPts, maxlen=4)
		players = deque(players, maxlen=4)
		courtPts, rawPts = sub_mapping(frame, tpl, num, 
					courtPts=courtPts, rawPts=rawPts, players=players, matchedKey=matchedKey)
		break
	with open(dst, "wb") as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		for (a,b), (c,d) in zip(courtPts, rawPts):
			writer.writerow([a,b,c,d])
	#cap.release()
	#cv2.destroyAllWindows()
	grid(args)

def neighbors(src, dstIdx, nbrs, d=5):

	srcIdx = np.vstack(np.nonzero(src)[::-1]).T
	distances, indices = nbrs.kneighbors(srcIdx)
	idx = distances<d
	nKeys = dstIdx[indices[idx]]
	oKeys = srcIdx[idx.ravel()]
	return oKeys, nKeys

def grid(args):
	src, dest, init = args['src'], args['grid'], args['dst']
	grid = dest + ".m.npy"
	dest = dest + ".avi"
	cap = cv2.VideoCapture(src)
	ret, frame = cap.read()
	frame_h, frame_w, _ = frame.shape
	Pts = np.genfromtxt(init, delimiter=",")
	courtPts, framePts = Pts[:, :2], Pts[:, -2:]
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(dest, fourcc, 20.0, (frame_w, frame_h))
	tpl = template()

	num = 0
	closing = denoise(frame)
	M = cv2.getPerspectiveTransform(np.float32(courtPts), np.float32(framePts))
	Ms = [M]
	cRot = cv2.warpPerspective(tpl, M, (frame_w, frame_h))
	raw, blank, cRotc, blank2 = frame.copy(), cRot.copy(), frame.copy(), cRot.copy()
	while(cap.isOpened()):
		
		ret, frame = cap.read()
		if not ret : break
		closing = denoise(frame)
		#cv2.imshow('raw', frame)
		dstIdx = np.vstack(np.nonzero(closing)[::-1]).T
		#import ipdb; ipdb.set_trace()
		nbrs = NearestNeighbors(n_neighbors=1,  algorithm = 'ball_tree').fit(dstIdx)
		converge = deque([], maxlen=5)
		cnt = True
		num_c = 0
		while cnt:
			cv2.warpPerspective(tpl, M, (frame_w, frame_h), dst = cRot)

			oKeys, nKeys = neighbors(cRot, dstIdx, nbrs, d=15)
			dm, ret = cv2.findHomography(oKeys, nKeys)
			M = np.dot( dm , M )
			cv2.bitwise_and(cRot, closing, dst=blank)
			score = np.sum(blank)/float(np.sum(cRot))
			converge.append(score)
			csum = np.sum(np.diff(np.array(converge)))
			if num_c > 500 or (csum < 0 and len(converge)>1):
				cnt = False
			num_c+=1
		Ms.append(M)
		cv2.cvtColor(cRot, cv2.COLOR_GRAY2BGR, dst=cRotc)
		cRotc[:,:, 0] = 0
		cRotc[:,:, 2] = 0
		cv2.bitwise_or(cRotc, frame, dst=raw)
		cv2.putText(raw, "# %s %s"%(num, num_c), (30,30), font, 1, (255,255,255), 1)
		cv2.putText(raw, "S %.2f %%"%(100*score), (30,70), font, 1, (255,255,255), 1)
		cv2.putText(raw, "D %.2f %%"%(100*csum), (30,110), font, 1, (255,255,255), 1)
		out.write(raw)
		#cRot = cv2.bitwise_not(cRot)
		#closing = cv2.bitwise_not(closing)
		cv2.imshow('frame', raw)
		k = cv2.waitKey(20) & 0xFF 
		if k == ord('c'):
			break
		elif k == ord('q'):
			cnt = False
			cap.release()
			break
		while False:
			cv2.imshow('frame', raw)
			cv2.imshow('crot', cRot)
			cv2.imshow('closing', closing)
			k = cv2.waitKey(20) & 0xFF 
			if k == ord('c'):
				break
			elif k == ord('q'):
				cnt = False
				cap.release()
				break
		num += 1
	out.release()
	cap.release()
	cv2.destroyAllWindows()
	if grid:
		np.save(grid, np.array(Ms))



def main():
	parser = argparse.ArgumentParser(description="perspective transform the given video data")
	parser.add_argument('-s', '--src', help='source video filename', required=True)
	parser.add_argument('-d', '--dst', help='case folder name', required=True)
	args = vars(parser.parse_args())
	if not os.path.exists(args['dst']):
		os.makedirs(args['dst'])
	args = vars(parser.parse_args())
	params = {'src' : args['src'], 
				"dst" : "%s/init.csv"%args['dst'],
				"grid" : "%s/grid"%args['dst']}

	mapping(params)

def demo():
	args = {'src' : "../data/heat3.avi", 
			"dst": "../data/heat3.1.csv",
			"grid" : "../data/heat3.1.grid"}
	mapping(args)
if __name__ == '__main__':
	main()