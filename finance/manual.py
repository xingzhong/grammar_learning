import numpy as np
import cv2
import glob
import csv

font = cv2.FONT_HERSHEY_PLAIN
Labels = ['Unknown', 'FW', 'RW', 'HS', 'IHS', 'RC', "FC", "BET", "BUT", "BER", "BUR", "BBOT", "BTOP"]

if __name__ == '__main__':
	cnt = True
	truths = []
	for fn in sorted(glob.glob("figs/*.png")):
		if not cnt: break
		img = cv2.imread(fn, cv2.IMREAD_COLOR)
		h,w = img.shape[:2]
		tpls = map(lambda x: cv2.imread("tpl/%s.png"%x, cv2.IMREAD_COLOR), Labels)
		tpls = map(lambda x:cv2.resize(x, (w,h)), tpls)
		blank = np.uint8(np.zeros((w, 2*h, 3)))
		#import ipdb; ipdb.set_trace()
		idx = int( fn.split('/')[-1][:-4] )
		labelId = 0
		while True:
			#imgcp = img.copy()
			blank[:h, :] = img
			blank[h:, :] = tpls[labelId]
			cv2.putText(blank, Labels[labelId], (10,20), font, 1,(0,0,0),1,cv2.LINE_AA)
			#cv2.addWeighted(imgcp,0.7,tpls[labelId],0.3,0, dst=imgcp)
			#cv2.imshow('image',imgcp)
			cv2.imshow('tpl', blank)
			k = cv2.waitKey(50) & 0xFF
			if k == ord('q'):         
				cv2.destroyAllWindows()
				cnt = False
				with open('truth.csv', 'wb') as fs:
					writer = csv.writer(fs)
					writer.writerows(truths) 
				break
			elif k == 32:
				print idx, Labels[labelId]
				truths.append([idx, Labels[labelId]])
				break
			elif k == ord('d'):
				labelId += 1
				if labelId > len(Labels)-1:
					labelId = 0
			elif k == ord('a'):
				labelId -= 1
				if labelId < 0:
					labelId = len(Labels)-1