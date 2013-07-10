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