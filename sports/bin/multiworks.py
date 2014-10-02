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
			pass

if __name__ == '__main__':
	#batch_tracking()
	batch_recognition()