import pprint 
import numpy
import operator
import sys
import networkx as nx
from bs4 import BeautifulSoup
from operator import attrgetter
import matplotlib.pyplot as plt


DEBUG = 0
merge_threshold = 200

class Place(object):
    def __init__(self, pos=(0,0,0,0), thing=set(), action=[]):
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

    def __str__(self):
        return 'Place ' + "%s at %s"%(self.thing, self.pos)
        
    def __add__(t1, t2, alpha=[0.5, 0.2, 0.3]):
        """
            overload "+" for merge two place objects
        """
        if not isinstance(t1, Place) or not isinstance(t2, Place):
            raise TypeError
        pos = (0,0,0,0)
        if t1.pos[2] == t2.pos[2] and t1.action == t2.action :
            # same time and same action
            pos = ((t1.pos[0] + t2.pos[0])/2.0, (t1.pos[1] + t2.pos[1])/2.0,
                     max(t1.pos[2], t2.pos[2]), min(t1.pos[3], t2.pos[3]))
           
            action = t1.action
            thing = t1.thing | t2.thing
            t3 = Place(pos, thing=thing, action=action)
            if DEBUG : pprint.pprint ({"t1":t1, "t2":t2, "t3":t3, "type":"thing merge"})
            return t3
            
        elif len(t1.thing & t2.thing) > 0 and t1.pos[2] != t2.pos[2]:
            # thing have common 
            #pos = max(t1.pos, t2.pos , key=lambda x: x[2])
            pos = ((t1.pos[0] + t2.pos[0])/2.0, (t1.pos[1] + t2.pos[1])/2.0,
                     min(t1.pos[2], t2.pos[2]), max(t1.pos[3], t2.pos[3]))
            thing = t1.thing & t2.thing
            action = t1.action.extend(t2.action)
            t3 = Place(pos, thing=thing, action=action)
            if DEBUG : pprint.pprint ({"t1":t1, "t2":t2, "t3":t3, "type":"action merge"})
            return t3
        elif t1.pos[3] > t2.pos[2] or t2.pos[3] > t1.pos[2]:
            pos = ((t1.pos[0] + t2.pos[0])/2.0, (t1.pos[1] + t2.pos[1])/2.0,
                   min(t1.pos[2], t2.pos[2]), max(t1.pos[3], t2.pos[3]))
            action = t1.action
            thing = t1.thing | t2.thing
            t3 = Place(pos, thing=thing, action=action)
            if DEBUG : pprint.pprint ({"t1":t1, "t2":t2, "t3":t3, "type":"thing merge"})
            return t3
        return False
    
    def __sub__(p1, p2):
        """
            overload "-" for find the metric distance 
        """
        if p1!=None and p2!=None:
            return numpy.sqrt((p1.pos[1]-p2.pos[1])**2+(p1.pos[0]-p2.pos[0])**2)
        else:
            print 'p1=', p1, 'p2=', p2
            return 0

def myMax( args ):
    max= args[0]
    for a in args[1:]:
        if a > max: max= a
    return max
def getMinStartTime( args ):
    min = args[0]
    for a in args[1:]:
        if a.pos[2] < min.pos[2]: min= a
    return min

#TODO:
#Try to use different way to detect group, such as social force model
#How to track individuals after two group get together, different factors
#Get grouping field by multiplying force and duration 
#Find VIP in a group


def merge_thing(data):
    #for each place object, merge into group by distance
    
    root = data['root']
    G = data['G']
    candidate = {}
    print 'Before merge, amount of nodes is ' + str(G.number_of_nodes())
    neighbors = G.neighbors(root)
    thing_merge_counter = 0
    action_merge_counter = 0
    
    #to record the frame windows
    min_start_time = 0
    max_end_time = 0
    
    #print len(neighbors)
    node_counter = 0
    minobj = neighbors[0]
    last_node= None
    
    while len(neighbors) > 1:        
        n1 = getMinStartTime(neighbors)

        #avoid repeat processing
        if n1 == last_node:
            neighbors.remove(n1)
            continue
        last_node = n1

        #set the window for Place nodes
        candidate[n1] = {}
        min_start_time = n1.pos[2] - 30
        if min_start_time < 0 :
            min_start_time = 0
        max_end_time = n1.pos[3] + 30
        
        #print 'n1 = ' + str(n1) + 'min_start_time = ' + str(min_start_time) + ' len = ' + str(len(neighbors)) 
        for n2 in neighbors:
            if n2.thing.issubset(n1.thing) or n2.thing == n1.thing:
                continue
            
            #if two places have commom time interval, append them into candidate
            if n1.pos[2] < n2.pos[3] and n2.pos[2] < n1.pos[3]:
                candidate[n1][n2] = n1 - n2
            elif G.successors(n2):
                if G.successors(n2)[0] > min_start_time and G.successors(n2)[0] < max_end_time:
                    #print 'First append ' + str(G.successors(n2)[0])
                    neighbors.append(G.successors(n2)[0])
                    neighbors.remove(n2)
                    node_counter = node_counter+1
                continue                
        
        #get the min candidate to do thing merge
        #if no candidate remove it and add succ
        if len(candidate[n1].keys()) == 0:
            if G.successors(n1):
                #print 'second append ' + str(n1)+ str(G.successors(n1)[0])
                neighbors.append(G.successors(n1)[0])
                node_counter = node_counter+1
            neighbors.remove(n1)
            continue
        min_key = min(candidate[n1], key=candidate[n1].get)
        n3 = min_key
            
        if n1 - n3 < merge_threshold:
                thing_merge_counter = thing_merge_counter + 1
                n4 = n1 + n3
                print str(n1) + str(n3) + ' will be merged into n4 ' + str(n4)
                G.add_node(n4)
                if G.successors(n1):
                    #print 'third append ' + str(G.successors(n1)[0])
                    #neighbors.append(G.successors(n1)[0])
                    G.add_edge(n4, G.successors(n1)[0])
                    G.remove_edge(n1, G.successors(n1)[0])
                    
                if G.predecessors(n1):
                    G.add_edge(G.predecessors(n1)[0], n4)
                    G.remove_edge(G.predecessors(n1)[0], n1)

                if G.successors(n3):
                    #print 'forth append ' + str(G.successors(n3)[0])
                    neighbors.append(G.successors(n3)[0])
                    G.add_edge(n4, G.successors(n3)[0])
                    G.remove_edge(n3, G.successors(n3)[0])
                    node_counter = node_counter+1
                    
                if G.predecessors(n3):
                    G.add_edge(G.predecessors(n3)[0], n4)
                    G.remove_edge(G.predecessors(n3)[0], n3)
                
                G.remove_node(n1)
                G.remove_node(n3)
                neighbors.append(n4)
                neighbors.remove(n1)
                neighbors.remove(n3)
                del candidate[n1]
                        
        '''
        successors = G.successors(n1)
        #if it is the end of the path, skip it
        if not successors:
            continue;
        
        #if no merge with other obj, do action merge with same obj
        while isMerged==0 and  G.successors(n1):
            #print 'n4 = ' + str(n1)
            #TODO: add filter
            #successors[0].pos[2] > min_start_time  and successors[0].pos[3] < max_end_time
            n5 = G.successors(n1)[0]
            if len(n1.thing & n5.thing) > 0 and n1.pos[2] <= n5.pos[3] or n5.pos[2] <= n1.pos[3] and n1 - n5 < merge_threshold:
                action_merge_counter = action_merge_counter + 1
                n4 = n1 + n5
                #print 'n4 second was added' + str(n4)
                G.add_node(n4)
                
                predes = G.predecessors(n1)
                if predes:
                    G.add_edge(predes[0], n4, color='green')
                succs = G.successors(n5)
                if succs:
                    G.add_edge(n4, succs[0], color='green')
                    G.remove_node(n1)
                    #neighbors.remove(n1)
                    print 'n5' + str(n5)
                    n1 = succs[0]
                    G.remove_node(n5)
                    neighbors.append(n4)
                else:
                    G.remove_node(n1)
                    G.remove_node(n5)
                    break
            '''
    print  str(node_counter) + ' nodes are precessed'                
    print 'There are ' + str(thing_merge_counter) + ' thing_merge and ' + str(action_merge_counter) + ' action_merge'
    print 'After merge, amount of nodes is ' + str(G.number_of_nodes())
    return G

def load_create(file):
    soup = BeautifulSoup(open(file))
    G=nx.DiGraph()

    root = Place(pos = (0, 0, 0, 0), thing = set())
    G.add_node(root)
    color_list = ['w', 'g', 'b', 'y', 'b']
    for obj in soup.find_all('object'):
        idnum = obj['id']
        leftneighbor = root
        for d in obj.find_all('data:bbox'): #[:20]
            p = Place(
                pos = ( int(d['x']), int(d['y']), int(d['framespan'].split(':')[0]), int(d['framespan'].split(':')[1])),
                thing = set([idnum])
                )
            #print str(p)
            G.add_node(p)
            #G.add_node(p, color = color_list[int(idnum)%5])            
            
            if  leftneighbor == root: 
                G.add_edge(root, p)
                leftneighbor = p
            else:
                G.add_edge(leftneighbor, p)
                leftneighbor = p
    return {'root':root, 'G':G }

def draw_graph(G):
    pos = {}
    color=nx.get_node_attributes(G,'color')
    
    grouplist = []
    singlelist = []
    for n in G.nodes():
        pos[n] = n.pos[:2]
        color[n] = nx.get_node_attributes(G,'color')
        if len(n.thing)>1:
            grouplist.append(n)
        else:
            singlelist.append(n)

    #nx.draw(G, with_labels=False,cmap=plt.cm.Greys,pos=pos,vmin=0,vmax=1)
        
    nx.draw_networkx_nodes(G,pos, nodelist=singlelist,
                       node_color='blue', node_size=20, alpha=0.8)
    nx.draw_networkx_nodes(G,pos, nodelist=grouplist,
                       node_color='red', node_size=400, alpha=0.1)
    nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
    
    # show graph
    plt.show()

def save_graph(G):
    #nx.write_gexf(G, "test.gexf")
    nx.write_graphml(G, "test.graphml",encoding ='UTF-8')
    
if  __name__ == '__main__':

    data = load_create('dataset/1-11200.xgtf')
    #G = data['G']
    #draw_graph(G)
    
    G = merge_thing(data)
    
    #save_graph(G)
    draw_graph(G)
    

