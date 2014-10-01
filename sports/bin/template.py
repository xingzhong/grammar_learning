import numpy as np
import cv2

CLength = 2865	# 94ft
CWidth = 1524	# 50ft
LWidth = 488	# 16ft
TLine = 724		# 23ft 9in
RLine = 183		# 6ft
FreeLine = 579	# 19ft
HOOP = 160 		# 4ft 15in
SPACE = 91		# 3ft
TSLine = 427	# 14ft
SideLine = 853	# 28ft
marginX = 100
marginY = 400

Pts1 = [[0,0], [CLength/2, 0], [CLength/2, CWidth], [0, CWidth], [CLength, CWidth], [CLength, 0], [0,0],[0, CWidth]]
Pts2 = [[0,518], [0, 1006], [FreeLine, 1006], [FreeLine, 518]]
Pts3 = [[CLength,518], [CLength, 1006], [CLength-FreeLine, 1006], [CLength-FreeLine, 518]]
Pts4 = [(HOOP, CWidth/2), (FreeLine, CWidth/2), (CLength-HOOP, CWidth/2), (CLength-FreeLine, CWidth/2) ]
Pts5 = [[0,579], [0, 945], [FreeLine, 945], [FreeLine, 579]]
Pts6 = [[CLength,579], [CLength, 945], [CLength-FreeLine, 945], [CLength-FreeLine, 579]]
Pts7 = [(0, SPACE), (TSLine, SPACE), (0,CWidth-SPACE), (TSLine, CWidth-SPACE)]
Pts8 = [(CLength, SPACE), (CLength-TSLine, SPACE), (CLength,CWidth-SPACE), (CLength-TSLine, CWidth-SPACE)]
Pts9 = [(SideLine, 0), (SideLine, CWidth), (CLength-SideLine, 0), (CLength-SideLine, CWidth)]
Pts10 = [(FreeLine-RLine, CWidth/2), (FreeLine+RLine, CWidth/2), (CLength-FreeLine-RLine, CWidth/2),
		 (CLength-FreeLine+RLine, CWidth/2), (HOOP+TLine, CWidth/2), (CLength-TLine-HOOP, CWidth/2),
		 (CLength/2, CWidth/2)]
Pts11 = [(TSLine, 0), (TSLine, CWidth), (CLength-TSLine, 0), (CLength-TSLine, CWidth),
			(0, 518-SPACE), (0, 1006+SPACE), (CLength, 518-SPACE), (CLength, 1006+SPACE)]

Pts1 = (np.array(Pts1, np.int32)+np.array([marginX, marginY])).reshape((-1,1,2))
Pts2 = (np.array(Pts2, np.int32)+np.array([marginX, marginY])).reshape((-1,1,2))
Pts3 = (np.array(Pts3, np.int32)+np.array([marginX, marginY])).reshape((-1,1,2))
Pts4 = np.array(Pts4, np.int32)+np.array([marginX, marginY])
Pts5 = (np.array(Pts5, np.int32)+np.array([marginX, marginY])).reshape((-1,1,2))
Pts6 = np.array(Pts6, np.int32)+np.array([marginX, marginY])
Pts7 = np.array(Pts7, np.int32)+np.array([marginX, marginY])
Pts8 = np.array(Pts8, np.int32)+np.array([marginX, marginY])
Pts9 = np.array(Pts9, np.int32)+np.array([marginX, marginY])
Pts10 = np.array(Pts10, np.int32)+np.array([marginX, marginY])
Pts11 = np.array(Pts11, np.int32)+np.array([marginX, marginY])


Bound = [(0,0), (CLength,0), (0,CWidth), (CLength, CWidth)]
Bound = (np.array(Bound, np.float32)+np.array([marginX, marginY])).reshape(-1,1,2)
Bound /= 2

POI = np.vstack((Pts1, Pts2, Pts3, Pts5)).reshape((-1,2))
POI = np.vstack((POI, Pts4, Pts6, Pts7, Pts8, Pts9, Pts10, Pts11))
POI /= 2

def visM():
	courtPts = Pts2
	framePts = [[100, 550], [0, 900], [550, 900], [550, 550]]
	M = cv2.getPerspectiveTransform(np.float32(courtPts), np.float32(framePts))
	return M

def template():
	tpl = np.uint8(np.zeros((2000, 3200)))
	tpl = cv2.polylines(tpl,[Pts1, Pts2, Pts3, Pts5, Pts6], True, 255, 5)
	tpl = cv2.ellipse(tpl, tuple(Pts4[0]), (TLine, TLine), 270, 21, 159, 255, 5)
	tpl = cv2.circle(tpl, tuple(Pts4[0]), 30, 255, 5)
	tpl = cv2.circle(tpl, tuple(Pts4[1]), RLine, 255, 5)
	tpl = cv2.ellipse(tpl, tuple(Pts4[2]), (TLine, TLine), 90, 21, 159, 255, 5)
	tpl = cv2.circle(tpl, tuple(Pts4[2]), 30, 255, 5)
	tpl = cv2.circle(tpl, tuple(Pts4[3]), RLine, 255, 5)
	tpl = cv2.line(tpl, tuple(Pts7[0]), tuple(Pts7[1]), 255, 5)
	tpl = cv2.line(tpl, tuple(Pts7[2]), tuple(Pts7[3]), 255, 5)
	tpl = cv2.line(tpl, tuple(Pts8[0]), tuple(Pts8[1]), 255, 5)
	tpl = cv2.line(tpl, tuple(Pts8[2]), tuple(Pts8[3]), 255, 5)
	tpl = cv2.line(tpl, tuple(Pts9[0]), tuple(Pts9[0]+np.array([0, SPACE])), 255, 5)
	tpl = cv2.line(tpl, tuple(Pts9[1]), tuple(Pts9[1]-np.array([0, SPACE])), 255, 5)
	tpl = cv2.line(tpl, tuple(Pts9[2]), tuple(Pts9[2]+np.array([0, SPACE])), 255, 5)
	tpl = cv2.line(tpl, tuple(Pts9[3]), tuple(Pts9[3]-np.array([0, SPACE])), 255, 5)
	#cv2.polylines(tpl,[Pts2], True, 255, 20)
	tpl = cv2.resize(tpl,None,fx=.5, fy=.5)
	return tpl

def maskTpl():
	tpl = np.uint8(np.zeros((2000, 3200, 3)))
	cv2.rectangle(tpl, 
			tuple(Pts1[0][0]-np.array([0,0])), 
			tuple(Pts1[4][0]+np.array([0,0])), (255,255,255), -1)
	#cv2.bitwise_not(tpl, dst=tpl)
	tpl = cv2.resize(tpl,None,fx=.5, fy=.5)
	return tpl

if __name__ == '__main__':
	tpl = template()
	mask = maskTpl()
	m = visM()
	visTpl = cv2.warpPerspective(tpl, m, (1280, 720))
	tpl = cv2.bitwise_not(tpl)
	while(True):
		
		cv2.imshow('template', tpl)
		cv2.imshow('visTpl', visTpl)
		k = cv2.waitKey(500) & 0xFF
		if k == 27: break
	cv2.destroyAllWindows()