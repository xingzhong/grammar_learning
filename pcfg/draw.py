import matplotlib
import matplotlib.pyplot as plt

def draw(s1, s2):
	fig = plt.figure()
	ax = fig.add_subplot(211)
	ax.plot(s1, "o-")
	ax.grid(True)
	bx = fig.add_subplot(212)
	bx.plot(s2, "o-")
	bx.grid(True)
	plt.show()

def draw2D(s):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(s[:,0], s[:,1], "o-")
	ax.grid(True)
	plt.show()

if __name__ == '__main__':
	import numpy as np
	sample = np.array([[1, 1], [2, 2]])
	print sample
	path = np.cumsum( sample , axis=0)
	print path
	draw2D(path)