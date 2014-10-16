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

def draw(keyFrames, epsF):
	
	import matplotlib.pyplot as plt
	import matplotlib.gridspec as gridspec
	import matplotlib as mpl
	#fig, axes = plt.subplots(ncols=4, figsize=(16,3))
	fig = plt.figure(figsize=(16,3))
	#gs = gridspec.GridSpec(2, 5, height_ratios=[15,1])
	gs = gridspec.GridSpec(1, 5)
	gs.update(left=0.05, right=0.95, top=1, bottom=0.0, wspace=0.02, hspace=0.0)
	#axe = plt.subplot(gs[1, :])
	
	axes = map(lambda x: plt.subplot(gs[0, x]), range(5))
	axe = fig.add_axes([0.05,0.12,0.9,0.06])
	framesIdx = map(lambda x: x[1][1], keyFrames)
	titles = map(lambda x: x[0], keyFrames)
	if 'defence' in titles:
		cmap = mpl.colors.ListedColormap(['b', 'g', 'm', 'c'])
	else:
		cmap = mpl.colors.ListedColormap(['b', 'g', 'm', 'r'])
	cmap.set_over('0.15')
	cmap.set_under('0.85')
	norm = mpl.colors.BoundaryNorm(framesIdx, cmap.N)
	cb2 = mpl.colorbar.ColorbarBase(axe, cmap=cmap,
                                     norm=norm,
                                     # to use 'extend', you must
                                     # specify two extra boundaries:
                                     boundaries=[-1]+framesIdx+[framesIdx[-1]+1],
                                     extend='both',
                                     #ticks=framesIdx, # optional
                                     spacing='proportional',
                                     orientation='horizontal')


	for ax, (title, (frame, number)) in zip(axes, keyFrames):
		#import ipdb ; ipdb.set_trace()
		ax.imshow(frame[:, :, ::-1])
		#ax.axis('off')
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_title("%s [#%s]"%(title, number))

	axe.axis('off')
	#axe.set_xlim((0, framesIdx[-1]))
	#axe.set_ylim((0, 1))
	#plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
	#plt.show()
	plt.savefig(epsF, format='eps', dpi=150)

def batch_vis():
	import pandas as pd
	import matplotlib.pyplot as plt
	parser = argparse.ArgumentParser(description="batch tracking")
	parser.add_argument('-s', '--src', help='data folder', required=True)
	parser.add_argument('-i', '--id', help='case id', required=True)
	args = vars(parser.parse_args())
	if args['src'][-1] != '/': args['src'] = args['src'] + "/"
	files = glob.glob(args['src']+"*_%s/"%args['id'])

	for fn in files:
		vidF = fn + 'detect.avi'
		decodeF = fn + 'decode.csv'
		epsF = fn[:-1] + ".eps"
		if os.path.isfile(vidF) and os.path.isfile(decodeF):
			print vidF, decodeF, epsF
			df = pd.read_csv(decodeF, header=None)
			idx = df.groupby(df[0]).first().sort(columns=1)[1]
			#import ipdb ; ipdb.set_trace()
			idx = idx.append(pd.Series({"end":len(df)-1}) )
			keyFrames = zip( idx.index.values , getFrame(vidF, idx.values))
			draw(keyFrames, epsF)

if __name__ == '__main__':
	#batch_tracking()
	#batch_recognition()
	batch_vis()
