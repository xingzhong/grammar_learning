import pprint 
import numpy
import operator
import sys
import networkx as nx
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt


DEBUG = None

class Place(object):
    def __init__(self, pos=(0,0,0, 0), thing=set(), action=[]):
        """
            Construct a Place object.
                pos itself should be represented as a 3/4D vector (x,y,[z],t)
                thing is an unordered set contains the entities or agents
                action is an ordered list that abstract the sequence movement, 
                     default is an empty action
        """
        if not action : action = [] 
        self.thing = thing
        self.pos = pos
        self.action = action
    def __del__(self):
        class_name = self.__class__.__name__
        #print class_name,'self refcount', sys.getrefcount(self), "destroyed"

    def __repr__(self):
        return "@%s do %s at %s"%(self.thing, self.action, self.pos)
        
    def __add__(t1, t2, alpha=[0.5, 0.2, 0.3]):
        """
            overload "+" for merge two pos objects
        """
        pos = (0,0,0)
        if t1.pos[2] == t2.pos[2] and t1.action == t2.action :
            # same time and same action
            pos = ((t1.pos[0] + t2.pos[0])/2.0, (t1.pos[1] + t2.pos[1])/2.0,
                     (t1.pos[2] + t2.pos[2])/2.0)
           
            action = t1.action
            thing = t1.thing | t2.thing
            t3 = Place(pos, thing=thing, action=action)
            if DEBUG : pprint.pprint ({"t1":t1, "t2":t2, "t3":t3, "type":"thing merge"})
            return t3
            
        elif len(t1.thing & t2.thing) > 0 and t1.pos[2] != t2.pos[2]:
            # thing have common 
            pos = max(t1.pos, t2.pos , key=lambda x: x[2])
            thing = t1.thing & t2.thing
            action = t1.action.extend(t2.action)
            t3 = Place(pos, thing=thing, action=action)
            if DEBUG : pprint.pprint ({"t1":t1, "t2":t2, "t3":t3, "type":"action merge"})
            return t3
            
        return None
    
    def __sub__(p1, p2):
        """
            overload "-" for find the metric distance 
        """
        if p1!=None and p2!=None:
            return numpy.sqrt((p1.pos[1]-p2.pos[1])**2+(p1.pos[0]-p2.pos[0])**2)
        else:
            print 'p1=', p1, 'p2=', p2
            return 0
            


def merge(data):
    
    #for each pos, do a possible merge and return G
    
    root = data['root']
    G = data['G']
    candidate = {}
    print G.number_of_nodes()
    neighbors = G.neighbors(root)
    
    for n1 in neighbors:
        candidate[n1] = {}
        for n2 in neighbors:
            #if two places have commom time line, appent them into candidate
            if n1.pos[2] < n2.pos[3] or n2.pos[2] < n1.pos[3]:
                candidate[n1][n2] = n1 - n2

            #get the min candidate to merge
            n3 = min(candidate[n1])
            if n1 - n3 < 10 and n1!=n3:
                    n4 = n1 + n3
                    G.add_node(n4)
                    G.add_edge(n4, G.succ(n1))
                    G.add_edge(n1, n4)
                    G.remove_node(n1)
                    G.remove_node(n3)
    return G

def load_create(file):
    soup = BeautifulSoup(open(file))
    G=nx.DiGraph()

    root = Place(pos = (0, 0, 0, 0), thing = set())
    G.add_node(root)
                
    for obj in soup.find_all('object'):
        idnum = obj['id']
        leftneighbor = root
        for d in obj.find_all('data:bbox'):
            p = Place(
                pos = ( int(d['x']), int(d['y']), int(d['framespan'].split(':')[0]), int(d['framespan'].split(':')[1])),
                thing = set([])
                )
            #print p
            if leftneighbor-p > 2 or leftneighbor == root:
                G.add_node(p)
                G.add_edge(leftneighbor, p)
                leftneighbor = p
    return {'root':root, 'G':G }

if  __name__ == '__main__':

    data = load_create('dataset/1-11200.xgtf')
    G = merge(data)
    
    # draw graph
    pos = {}
    for n in G.nodes():
        pos[n] = n.pos[:2]
    nx.draw(G,node_color='r', with_labels=False,cmap=plt.cm.Greys,pos=pos,vmin=0,vmax=1)
    
    # show graph
    plt.show()



