from bs4 import BeautifulSoup
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from mayavi import mlab

class Behave(object):
	def __init__(self, attrs, idnum):
	# thing is someone, 
	# place is sometime/place (space-time 3D), 
	# action is do something
		
		time = map(int, attrs['framespan'].split(':'))
		self.thing = int(idnum)
		self.place = (int(attrs['x']), 
			int(attrs['y']), tuple(time))
		self.action = -1
	
	def __str__(self):
		return "[%s] : %s"%(self.thing, self.place)

def to4D(bs):
	for d in bs:
		for z in range(d.place[2][0], d.place[2][1]+1) :
			for i in (d.place[0], d.place[1], z, d.thing):
				yield i

	
def visual(num, dm, line):
	index = np.nonzero(dm[:,2] == num)[0]
	
	for ind in index :	
		x,y,z,i = dm[ind, :]
		print x,y,z,i
		line.set_data(x, y)
		return line
	

if __name__ == '__main__':
	soup = BeautifulSoup(open('1-11200.xgtf'))
	data = []
	for obj in soup.find_all('object'):
		idnum = obj['id']
		for d in obj.find_all('data:bbox'):
			data.append(Behave( d.attrs, idnum ))
	
	iterator = to4D(data)
	dm = np.fromiter(iterator, np.int)
	dm = np.reshape(dm, (-1,4))
	
	
	dm = np.asanyarray(sorted(dm, key=lambda x: x[2]))
	frame = dm[-1, 2]
	
	print 
	
	fig = plt.figure()
	plt.xlim(0, 1000)
	plt.ylim(0, 1000)
	plt.grid()
	l, = plt.plot([],[],'o')
	
	ani = animation.FuncAnimation(fig, visual, frame, fargs=(dm,l), interval=40)
	plt.show()
	

	
	
