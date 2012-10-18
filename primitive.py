import pprint 
import numpy
import operator
from parsing import load

DEBUG = None

class Place(object):
    def __init__(self, place, thing=set(), action=[]):
        """
            Construct a Place object.
                place itself should be represented as a 3/4D vector (x,y,[z],t)
                thing is an unordered set contains the entities or agents
                action is an ordered list that abstract the sequence movement, 
                     default is an empty action
        """
        if not action : action = [] 
        self.thing = thing
        self.place = numpy.array(place)
        self.action = action
    def __repr__(self):
        return "@%s do %s at %s"%(self.thing, self.action, self.place)
        
    def __add__(t1, t2, alpha=[0.5, 0.2, 0.3]):
        """
            overload "+" for merge two place objects
        """
        if t1.place[2] == t2.place[2] and t1.action == t2.action :
            # same time and same action
            place = (t1.place + t2.place)/2.0
            action = t1.action
            thing = t1.thing | t2.thing
            t3 = Place(place, thing=thing, action=action)
            if DEBUG : pprint.pprint ({"t1":t1, "t2":t2, "t3":t3, "type":"thing merge"})
            return t3
            
        elif len(t1.thing & t2.thing) > 0 and t1.place[2] != t2.place[2]:
            # thing have common 
            place = max(t1.place, t2.place , key=lambda x: x[2])
            thing = t1.thing & t2.thing
            action = t1.action.extend(t2.action)
            t3 = Place(place, thing=thing, action=action)
            if DEBUG : pprint.pprint ({"t1":t1, "t2":t2, "t3":t3, "type":"action merge"})
            return t3
            
        return None
    
    def __sub__(t1, t2):
        """
            overload "-" for find the metric distance 
        """
        return numpy.linalg.norm(t1.place - t2.place)
            

def merge(places):
    """
    for each place, do a possible merge and return new places
    """
    candidate = {}
    itersets = list(places)
    for p1 in itersets:
        if p1 not in places :
            continue
        candidate[p1] = {}
        for p2 in places:
            d = p1 - p2
            if d:
                candidate[p1][p2] = d
        p3 = min(candidate[p1], key= lambda x: candidate[p1].get(x))
        if p1 - p3 < 20:
            p4 = p1 + p3
            places.remove(p1)
            places.remove(p3)
            places.add(p4)
                
    return places
    
if  __name__ == '__main__':
    print "hello Xingzhong"
    data = load('dataset/1-11200.xgtf')

    places = set()
    for d in data[:50] :
        for t in eval("range(%s+1)"%d[0]['framespan'].replace(":",",")):
            p = Place(
                place = ( int(d[0]['x']), int(d[0]['y']), t ),
                thing = set([int(d[1])]), 
                )
            places.add( p ) 
            
    places = merge(places)
    pprint.pprint (places) 


    