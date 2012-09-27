import pprint 
import numpy
import operator
from parsing import load


class Token(object):
    def __init__(self, thing=None, place=None, action=None):
        self.thing = thing
        self.place = numpy.array(place)
        self.action = action
    def __repr__(self):
        return "%s do %s at %s"%(self.thing, self.action, self.place)
        
    def __and__(t1, t2, alpha=[0.5, 0.2, 0.3]):
        p1 = t1.thing == t2.thing and 0 or 1
        p2 = t1.action == t2.action and 0 or 1
        p3 = numpy.linalg.norm(t1.place[:2] - t2.place[:2])
        return numpy.dot([p1, p2, p3], alpha)
    
    
if  __name__ == '__main__':
    print "hello Xingzhong"
    data = load('dataset/1-11200.xgtf')
    tokens = []
    for d in data :
        token = Token(
            thing = int(d[1]), 
            place = ( int(d[0]['x']), int(d[0]['y']) ) + tuple(map(int, d[0]['framespan'].split(":")))
            )
        tokens.append( token ) 
        
    pprint.pprint( tokens[1:3] )
    diff = map(operator.and_, tokens[:-1], tokens[1:]) 
    pprint.pprint( diff[:10] )
    
    