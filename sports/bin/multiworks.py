import glob
import argparse
import os
from grid import mapping, grid
from tracking import tracking_wrapper
from recognition import wrapper_recognition

def batch_grid():
	parser = argparse.ArgumentParser(description="multi grid works")
	parser.add_argument('-s', '--src', help='data folder', required=True)
	parser.add_argument('-i', '--id', help='case id', required=True)
	args = vars(parser.parse_args())
	if args['src'][-1] != '/': args['src'] = args['src'] + "/"

	for fn in glob.glob(args['src']+"*.avi"):
		name = args['src'] + os.path.basename(fn).split('.')[0] + "_%s"%args['id']
		if not os.path.exists(name): os.makedirs(name)
		params = {'src' : fn, 
				"dst" : "%s/init.csv"%name,
				"grid" : "%s/grid"%name}
		#print params
		if not os.path.isfile(params['dst']):	#init already existed
			mapping(params)

	for fn in glob.glob(args['src']+"*.avi"):
		name = args['src'] + os.path.basename(fn).split('.')[0] + "_%s"%args['id']
		params = {'src' : fn, 
				"dst" : "%s/init.csv"%name,
				"grid" : "%s/grid"%name}
		grid(params)

def batch_tracking():
	parser = argparse.ArgumentParser(description="batch tracking")
	parser.add_argument('-s', '--src', help='data folder', required=True)
	parser.add_argument('-i', '--id', help='case id', required=True)
	args = vars(parser.parse_args())
	if args['src'][-1] != '/': args['src'] = args['src'] + "/"
	for fn in glob.glob(args['src']+"*_%s/"%args['id']):
		print fn
		tracking_wrapper(fn)

def batch_recognition():
	parser = argparse.ArgumentParser(description="batch tracking")
	parser.add_argument('-s', '--src', help='data folder', required=True)
	parser.add_argument('-i', '--id', help='case id', required=True)
	args = vars(parser.parse_args())
	if args['src'][-1] != '/': args['src'] = args['src'] + "/"
	for fn in glob.glob(args['src']+"*_%s/"%args['id']):
		print fn
		try: 
			wrapper_recognition(fn)
		except RuntimeError:
			print "error"
			continue

def getFrame(vf, frames):
	import cv2
	cap = cv2.VideoCapture(vf)
	num = 0
	results = []
	while(cap.isOpened()):
		ret, frame = cap.read()
		if not ret : break
		if num in frames:
			results.append((frame, num))
		num+=1
	return results

def draw(keyFrames, fn):
	import matplotlib.pyplot as plt
	#import ipdb ; ipdb.set_trace()
	fig, axes = plt.subplots(ncols=len(keyFrames), figsize=(16, 2))
	for ax, (title, (frame, number)) in zip(axes, keyFrames):
		ax.imshow(frame[:, :, ::-1])
		ax.axis('off')
		ax.set_title("%s [#%s]"%(title, number))
		
	#plt.show()
	print fn
	plt.savefig(fn, format='eps', dpi=150)

def batch_vis():
	import pandas as pd
	
	parser = argparse.ArgumentParser(description="batch tracking")
	parser.add_argument('-s', '--src', help='data folder', required=True)
	parser.add_argument('-i', '--id', help='case id', required=True)
	args = vars(parser.parse_args())
	if args['src'][-1] != '/': args['src'] = args['src'] + "/"
	for fn in glob.glob(args['src']+"*_%s/"%args['id']):
		vidF = fn + 'detect.avi'
		decodeF = fn + 'decode.csv'
		epsF = fn[:-1] + ".eps"
		if os.path.isfile(vidF) and os.path.isfile(decodeF):
			print vidF, decodeF, epsF
			df = pd.read_csv(decodeF, header=None)
			idx = df.groupby(df[0]).first().sort(columns=1)[1]
			keyFrames = zip( idx.index.values , getFrame(vidF, idx.values))
			#draw(keyFrames, epsF)

if __name__ == '__main__':
	#batch_tracking()
	#batch_recognition()
	batch_vis()
