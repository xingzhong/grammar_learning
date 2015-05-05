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

def draw(keyFrames, framesIdx, epsF):
	
	import matplotlib.pyplot as plt
	import matplotlib.gridspec as gridspec
	import matplotlib as mpl
	fig = plt.figure(figsize=(16,3))
	n = 8
	gs = gridspec.GridSpec(1, n)
	gs.update(left=0.05, right=0.95, top=1, bottom=0.0, wspace=0.02, hspace=0.0)

	axes = map(lambda x: plt.subplot(gs[0, x]), range(n))
	axe = fig.add_axes([0.05,0.12,0.9,0.06])
	
	titles = ["Nothing", "picker", "block", 'defence', 'picker', 'block', 'defence', 'end']
	if 'defence' in titles:
		cmap = mpl.colors.ListedColormap(['b', 'g', 'm', 'c', 'g', 'm', 'c'])
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


	for ax, title, (frame, number) in zip(axes, titles, keyFrames):
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

if __name__ == '__main__':
	idx = [0, 10, 60, 100, 171, 200, 255, 445]
	frames = getFrame("../icip/heat20.avi", idx)
	draw(frames, idx, "../icip/heat20.eps")