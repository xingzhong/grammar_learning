import pandas as pd
import numpy as np
import networkx as nx
from event import drawG2, Event, EventGraph

AGENTS = ["HIP_CENTER" , "SPINE" , "SHOULDER_CENTER", "HEAD", "SHOULDER_LEFT", "ELBOW_LEFT", "WRIST_LEFT", "HAND_LEFT",
 		"SHOULDER_RIGHT", "ELBOW_RIGHT", "WRIST_RIGHT", "HAND_RIGHT", "HIP_LEFT", "KNEE_LEFT", "ANKLE_LEFT", "FOOT_LEFT",
 		"HIP_RIGHT", "KNEE_RIGHT", "ANKLE_RIGHT", "FOOT_RIGHT"]

def instance_generator (panel, truth):
    old = 0
    for t, label in truth:
        sample = []
        for agent in AGENTS:
            sample.append( panel[agent].loc[old:t][['dx','dy','dz']].values )
        old = t
        yield np.array( sample )

def kinetic(fileName='P2_1_9_p07', M=None, N=None):
	FILE  = "/home/xingzhong/MicrosoftGestureDataset-RC/data/%s"%fileName
	truth = np.genfromtxt(FILE+'.tagstream', delimiter=';', skiprows=1, dtype=None, converters={0: lambda x: (int(x) *1000 + 49875/2)/49875})
	nd = np.loadtxt(FILE+'.csv')
	nd = nd[np.where(nd[:,80]!=0)]# remove empty rows
	idx, ndd = map(int, nd[:,0]), nd[:, 1:] # unpack index and data
	m, n = ndd.shape
	panel = pd.Panel( ndd.reshape((m, 20, 4)), items=idx, major_axis=AGENTS, minor_axis=['x','y','z','v'] ).transpose(2, 0, 1)
	panel['dx'] = 1000* panel['x'].diff().fillna(0)
	panel['dy'] = 1000* panel['y'].diff().fillna(0)
	panel['dz'] = 1000* panel['z'].diff().fillna(0)
	panel = panel.transpose(2, 1, 0)
	samples =  [s for s in instance_generator(panel, truth)] 
	g = EventGraph()

	for aid, seq in enumerate (samples[0]):
		if M is not None and aid > M :
			break
		for t, atom in enumerate (seq):
			if N is not None and t > N:
				break
			elif not atom is None and t!=0:

				g.addEvent( Event(t, aid, atom ))

	g.buildEdges(delta = 1)
	print nx.info(g)
	return g

if __name__ == '__main__':
	g = kinetic(M=0, N=10)
	drawG2(g, node_size=1600, cluster=True, label=True)