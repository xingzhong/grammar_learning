import numpy as np
D = np.random.randint(0, 20, (4,4))
D = np.reshape( range(16), (4,4))
D = np.array([[2,8,1,5],[4,16,2,3],[1,2,2,19],[3,5,2,1]])
print D
alpha_1 = np.array( [[2, 0, 0, 0],[0,1,0,0]] )
alpha_2 = np.array( [[4, 0, 0, 0],[0,1.2,0,0]] )
d1 = np.dot(alpha_1,  D)
d2 = np.dot(d1, alpha_2.T)

print d1
print d2
print np.var(d2)

