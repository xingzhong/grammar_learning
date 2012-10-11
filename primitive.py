import pprint 
import numpy
import operator
from parsing import load

DEBUG = 1

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
            
    
    
if  __name__ == '__main__':
    print "hello Xingzhong"
    data = load('dataset/1-11200.xgtf')

    places = []
    for d in data :
        for t in eval("range(%s+1)"%d[0]['framespan'].replace(":",",")):
            p = Place(
                place = ( int(d[0]['x']), int(d[0]['y']), t ),
                thing = set([int(d[1])]), 
                )
            places.append( p ) 
            
    
    places = sorted(places, key=lambda x: x.place[2])
    #pprint.pprint (places[:50]) 
    for p1 in places[:10] :
        for p2 in places[:10] :
            p3 = p1+p2
            if p3:
                print p3
    
    

    