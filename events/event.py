import networkx as nx
from itertools import combinations
from scipy.stats import norm
from sklearn.mixture import log_multivariate_normal_density
from sklearn.cluster import spectral_clustering
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint 
from scipy.sparse.linalg import eigs
from scipy.stats import norm


class Grammar():
    def __init__(self):
        self._rules = {}
    def addRule(self, lhs, rhs1, rhs2, op):
        if self._rules.has_key((lhs, op)):
            print "overwrite ", self._rules[(lhs, op)] 
        self._rules[(lhs, op)] = (lhs, rhs1, rhs2, op)
        print "new rule", [lhs, rhs1, rhs2, op]
    def fromEvent(self, c, a, b, d):
        # c -> a -[d]- b
        lhs = Event(-1, c._aids, c._semantics)
        rhs1 = Event(-1, a._aids, a._semantics)
        rhs2 = Event(-1, b._aids, b._semantics)
        op = d['type']
        self.addRule(lhs, rhs1, rhs2, op)
        return self
    
    def iterNT(self):
        for rule in self._rules.keys():
            yield rule

    def __repr__(self):
        string = ""
        for key, rule in self._rules.iteritems():
            string += str( key ) + str(rule) + str( rule[0]._semantics ) + "\n"
        return string

class Event():
    def __init__(self, time, aid, semantics):
        self._tp = time
        if isinstance(aid, frozenset):
            self._aids = aid
        else:
            self._aids = frozenset({aid})
        self._semantics = semantics
        
    def __sub__(self, other ):
        # define the metrics function \delta in events space
        # first assume seamntics are linear space
        
        return np.linalg.norm ( self._semantics - other._semantics )
    
    def __mul__(self, other):
        return np.dot(self._semantics, other._semantics)
    
    def __div__(self, other):
        dx = np.linalg.norm ( self._semantics - other._semantics )
        #print dx, np.exp(-(dx)**2)
        sigma = 1
        return np.exp(-sigma * (dx)**2)

    def dup__div__(self, other):
        x = np.array([self._semantics] )
        means = np.array([other._semantics])
        m,n = x.shape
        covars = np.ones((1,n))
        #print x
        #print means
        #print covars
        return log_multivariate_normal_density(x, means, covars)
    
    #def __add__(self, other):
        # define the merge function in events space
    #    pass
    
    def __repr__(self):
        string = "(%s)@%s"%(",".join(map(str, self._aids)), self._tp)
        return string
    
    def __hash__(self):
        return hash(self._tp) + hash(self._aids) + hash(str(self._semantics))
    
    def __eq__(self, other):
        return hash(self) == hash(other)
    
    @staticmethod
    def random():
        time = np.random.randint(1, 10)
        aid = np.random.randint(1, 5)
        semantics = np.random.normal(0, 1, (5, 1))
        return Event(time, aid, semantics)
    
    def likelihood (self, e1, e2):
        # P(e1, e2 | self) = \delta(self, e1) * \delta(self, e2)
        if e1._aids | e2._aids <= self._aids and e1._aids | e2._aids >= self._aids:
            #print e1._semantics, e2._semantics, self._semantics
            #return 1 / ( (self - e1 + 0.1) * (self - e2 + 0.1))
            return np.exp( (e1/self) + (e2/self) )
        else:
            return -np.inf
    
    @staticmethod
    def argMax(e1, e2):
        # best events abstraction in space
        # return a new event 
        time = min(e1._tp, e2._tp)
        aid = e1._aids | e2._aids
        semantics = ( e1._semantics + e2._semantics ) / 2
        #print "\t"*3, e1, e1._semantics
        #print "\t"*3, e2, e2._semantics
        #print "\t"*3, semantics
        return Event(time, aid, semantics)
    
    @staticmethod
    def decision(e1, e2, grammar, graph):
        # based on posterior probability, make merge decision
        bestPost, best = -np.inf, None
        for (nt, op) in grammar.iterNT():
            #post = graph.prior(nt) * nt.likelihood(e1, e2)
            post = Event.posterior(e1, e2, nt, graph)
            if post > bestPost:
                bestPost = post
                best = nt
        #print "\t\t", bestPost
        c = Event.argMax(e1, e2)
        #post = graph.prior(c) * c.likelihood(e1, e2)
        post = Event.posterior(e1, e2, c, graph)
        #print "\t\t\t", post, c
        if post > bestPost:
            bestPost = post
            best = c
        c = None
        #post = graph.prior(c) * np.exp((e1/e1) + (e2/e2))
        post = Event.posterior(e1, e2, c, graph)
        if post > bestPost:
            bestPost = post
            best = c
        #print "\t\t", bestPost, best
        return bestPost, best
    
    @staticmethod
    def posterior(e1, e2, c, graph):
        if c == None:
            return graph.prior(c) * np.exp((e1/e1) + (e2/e2))
        else:
            return graph.prior(c) * c.likelihood(e1, e2)

def learning(graph, grammar):
    for _ in range(2):
        grammar = graph.merge(grammar)
        graph = graph.update(grammar)
    return graph, grammar

def formartSemantic(x):
    return "\n".join(map( lambda x:"%.2f"%x, x.flat))

def drawG2(G, node_size=800, figsize=(18,8) , label=True ,edge=True, cluster=True, weight='delta', output=False, title=None):
    plt.figure(figsize=figsize)
    edge1=[(u,v) for (u,v,d) in G.edges(data=True) if d['type'] == "="]
    edge2=[(u,v) for (u,v,d) in G.edges(data=True) if d['type'] == "<"]
    #edge3=[(u,v) for (u,v,d) in G.edges(data=True) if d['type'] == ">"]
    edge_labels = { (u,v) : "%.2f"%d[weight] for (u,v,d) in G.edges(data=True) }
    #pprint (edge_labels)
    #labels = {n:str(d) for (n,d) in G.nodes(data=True)}
    initPos = { n:(n._tp , np.mean( map(float, n._aids ) )  ) 
        for (n,d) in G.nodes(data=True)}
    pos=nx.spring_layout(G, pos=initPos)
    
        
    labels = { n : formartSemantic(n._semantics) for n in G.nodes() }
    #nx.draw_networkx_nodes(G,pos=initPos,nodelist=node1, node_size=node_size, alpha=0.8, node_color='green')
    if cluster:
        colors = [d.get('cluster', 'white') for (n,d) in G.nodes(data=True)]
    else:
        colors = 'white'
    nx.draw_networkx_nodes(G,pos=initPos, node_size=node_size, alpha=0.8, node_color=colors, cmap=plt.get_cmap('Accent'))
    if edge:
        nx.draw_networkx_edges(G,pos=initPos,edgelist=edge1,
                        width=2, alpha=0.8, arrows=False)
        nx.draw_networkx_edges(G,pos=initPos,edgelist=edge2,
                        width=1, alpha=0.8, edge_color='b', arrows=True)
        #nx.draw_networkx_edges(G,pos=initPos,edgelist=edge3,
        #                width=1, alpha=0.8, edge_color='r', arrows=True)
    if label:
        nx.draw_networkx_labels(G,pos=initPos, labels = labels, font_size=12,font_family='sans-serif',label_pos=0.8)
        nx.draw_networkx_edge_labels(G, pos=initPos, edge_labels=edge_labels, font_size=10,font_family='sans-serif', alpha=0.5)
    #plt.axis('on')
    plt.grid()
    plt.xlabel("Time Grid")
    plt.ylabel("Agents")
    if title :
        plt.title(title)
    
    if output:
        plt.savefig("%s.eps"%output, dpi=1000)
    else:
        plt.show()
                
def drawG(G, node_size=800, figsize=(18,8)):
    plt.figure(figsize=figsize)
    edge1=[(u,v) for (u,v,d) in G.edges(data=True) if d['type'] == "="]
    edge2=[(u,v) for (u,v,d) in G.edges(data=True) if d['type'] == "<"]
    edge_labels = { (u,v) : "%.2f"%d['delta'] for (u,v,d) in G.edges(data=True) }
    #labels = {n:str(d) for (n,d) in G.nodes(data=True)}
    initPos = { n:(n._tp , np.mean( map(float, n._aids ) ) + np.random.normal(0, 0.1) ) 
        for (n,d) in G.nodes(data=True)}
    pos=nx.spring_layout(G, pos=initPos)
    node1=[n for (n,d) in G.nodes(data=True) if d['type'] == "NT"]
    node2=[n for (n,d) in G.nodes(data=True) if d['type'] == "T"]
    labels = { n : formartSemantic(n._semantics) for n in G.nodes() }
    nx.draw_networkx_nodes(G,pos=initPos,nodelist=node1, node_size=node_size, alpha=0.8, node_color='green')
    nx.draw_networkx_nodes(G,pos=initPos,nodelist=node2, node_size=node_size, alpha=0.8, node_color='white')
    nx.draw_networkx_edges(G,pos=initPos,edgelist=edge1,
                        width=2, alpha=0.8, arrows=False)
    nx.draw_networkx_edges(G,pos=initPos,edgelist=edge2,
                        width=2, alpha=0.8, edge_color='b', arrows=True)
    nx.draw_networkx_labels(G,pos=initPos, labels = labels, font_size=8,font_family='sans-serif')
    nx.draw_networkx_edge_labels(G, pos=initPos, edge_labels=edge_labels, font_size=10,font_family='sans-serif')
    plt.axis('on')
    plt.grid()
    plt.xlabel("Time Grid")
    plt.ylabel("Agents")
    plt.show()
    #plt.savefig("test.eps", dpi=1000)

class EventGraph(nx.DiGraph):
    def clustering(self, weight='delta', k=10):
        #print g.nodes()
        A = nx.adjacency_matrix(g, weight=weight)
        m, n = A.shape
        B = np.zeros_like(A)
        for i in range(m):
            for j in range(n):
                B[i, j] = max( A[i,j], A[j, i] )
        #print B
        #D = np.diag( np.sum(A, axis=1).flat )
        #D = np.diag( np.sum(B, axis=1).flat )
        #Dh =  np.diag( np.power( np.sum(B, axis=1), -0.5).flat )

        #L = Dh * (D-A) * Dh
        #L = Dh * (D-B) * Dh
        #L = D - A
        #L = D - B
        #w,v = np.linalg.eig(L)
        #idx = w.argsort()
        #w = w[idx]
        #v = v[:, idx]
        #nx.set_node_attributes(g, "cluster", dict(zip(g.nodes(), v[:,1]>0)))
        #return w, v
        #pprint( self.edges(data=True) )
        #print nx.adjacency_matrix(self, weight='delta')
        #L = nx.laplacian_matrix(self, weight='delta')
        #w, v = eigs(L, k=2 )
        #print w
        #print v
        #pprint (v[:,1])
        #pprint (self.nodes(data=True))
        #nx.set_node_attributes(self, "cluster", dict(zip(self.nodes(), v[:,1]>0)))
        #pprint (self.nodes(data=True))
        #A = nx.adjacency_matrix(self, weight='delta')
        #print A
        labels = spectral_clustering(B , n_clusters=k)
        #pprint (labels)
        nx.set_node_attributes(self, "cluster", dict(zip(self.nodes(), labels)))
        return labels

    def entropy(self):
        # calculate entropy 
        e = np.zeros((1,1))
        for (u,v,x) in self.edges(data=True):
            #print u,v,x, u/v, v/u, e
            if x['type'] == "=":
                e += xlog(np.exp(u/v))+xlog(np.exp(u/v))
            else:
                e += xlog(np.exp(u/v))
        return -e

    def update(self, grammar):
        #print self.edges(data=True)
        for (u,v,x) in self.edges(data=True):
            for (key, op), rule in grammar._rules.iteritems():
                if x['type'] == op:
                    #print "\t", u,v, key.likelihood(u, v)
                    #print "\t", u, v, Event.posterior(u, v, key, self), Event.posterior(u, v, None, self)
                    if Event.posterior(u, v, key, self) > Event.posterior(u, v, None, self):
                        self._merge(u, v, key, x)
                        #drawG(self)
                        return self.update(grammar)
        return None
    
    def merge(self, grammar):
        # argmax_{a, b, c} P(C | a, b, G)
        bestPost, best, a, b, d = -np.inf, None, None, None, None
        for (u,v,x) in self.edges(data=True):
            post, c = Event.decision(u, v, grammar, self)
            #print "\t", post, c, u, v
            if post > bestPost:
                bestPost = post
                best = c
                a = u
                b = v
                d = x
        if best:
            print "decision", bestPost, best, a, b
            return grammar.fromEvent (best, a, b, d)
        else:
            return best
    
    def _merge(self, a, b, nt, d):
        # merge a and b together to c
        # d is edge data
        
        c = Event(min(a._tp, b._tp), nt._aids, nt._semantics)
        self.add_node(c, type='NT')
        if d['type'] == '=':
            for (u,v,x) in self.in_edges([a, b], data=True):
                if not u == c:
                    self.add_edge(u, c, **x)
            for (u,v,x) in self.out_edges([a, b], data=True):
                if not c == v:
                    self.add_edge(c, v, **x)
        elif d['type'] == '<':
            for (u,v,x) in self.in_edges([a], data=True): 
                self.add_edge(u, c, **x)
            for (u,v,x) in self.out_edges([a], data=True):
                self.add_edge(c, v, **x)
            for (u,v,x) in self.out_edges([b], data=True):
                if x['type'] == "<":
                    self.add_edge(c, v, **x)
        
        #if c == b and a == c:
        #    pass
        if a == c:
            self.remove_nodes_from([b])
        else:
            self.remove_nodes_from([a, b])
            
    def prior(self, event):
        # P(C) supposed to be determined through data graph
        if event is None:
            return 0.25
        elif event in self:
            # naive assume 
            return 0.4
        else :
            return 0.35
    
    def addEvent(self, event):
        self.add_node(event, type='T')
    
    def timeIndex(self):
        return reduce(lambda x, y: x.union(set([y._tp])), self.nodes(), set())
        #return sorted( list(idx) )
    
    def agentIndex(self):
        return reduce(lambda x, y: x.union(y._aids), self.nodes(), set())
        
    def filterNodes(self, selector):
        return filter (selector , self.nodes())
    
    def buildEdges(self, delta = 0):
        # when all the events are available in graph,
        # the edge should be added through their ">" and "=" order.
        # EventSameTime
        for t in self.timeIndex():
            self._addEqualEdges( self.filterNodes(lambda x:x._tp == t))
        
        for aid in self.agentIndex():
            self._addPathEdges( self.filterNodes(lambda x: aid in x._aids) , delta)
        
    def _addEqualEdges(self, nodes):
        for x, y in combinations(nodes, 2):
            if hash(x) != hash(y) :
                self.add_edge( x, y, type='=', delta=x/y)
                self.add_edge( y, x, type='=', delta=y/x)
    
    def _addPathEdges(self, nodes, delta):
        sortedNodes = sorted( nodes, key=lambda x: x._tp)
        for i in range(len(sortedNodes)-1):
            if sortedNodes[i+1]._tp - sortedNodes[i]._tp <= delta:
                self.add_edge(sortedNodes[i], sortedNodes[i+1], type='<', delta=sortedNodes[i]/sortedNodes[i+1])
                #self.add_edge(sortedNodes[i+1], sortedNodes[i], type='>', delta=sortedNodes[i+1]/sortedNodes[i])

def learning (g):
    # given graph g, 
    # return grammar 
    gr = Grammar()
    while(True):
        numNodes = len(g)
        drawG(g, figsize=(18,4))
        if g.merge(gr):
            g.update(gr)
            if len(g) == numNodes:
                print "no more update"
                break
        else:
            print "no more"
            break
    return gr

if __name__ == '__main__':
    g = EventGraph()
    left = np.array([1.0, 0])
    right = np.array([-1.0, 0])
    up = np.array([0, 1.0])
    down = np.array([0, -1.0])
    sample = [[up, down, left, None] * 3,
                [down, up, right, None] * 3,
                [left, None, down, down] * 3 ]
    #sample = np.random.choice( [up, down, left, right, None], size=(3,5) )
    for aid, seq in enumerate (sample):
        for t, atom in enumerate (seq):
            if not atom is None:
                g.addEvent( Event(t, aid, atom+0.01*np.random.normal() ))

    g.buildEdges(delta = 1)
    
    drawG(g, figsize=(10,6))
    #g.clustering()
    #drawG2(g, node_size=1500, figsize=(20,10), label=True, edge=True, weight='delta', cluster=False)
    # print learning(g)
    S = []
    for (x,y,d) in g.edges(data=True):
        S.append(np.hstack((x._semantics, y._semantics)))
    S = np.array(S)
    from denseGaussian import estimate
    res, sels = estimate(S)
    print 'grammar', res[:2], res[2:]   
    nx.set_edge_attributes(g, "sel", dict(zip(g.edges(), sels)))
    nx.set_node_attributes(g, "cluster", dict(zip(g.nodes(), ['white']*len(g.nodes()))) )
    cluster = {}
    for (x,y,d) in g.edges(data=True):
        if d['sel']:
            cluster[x] = "red"
            cluster[y] = "blue"
    nx.set_node_attributes(g, "cluster", cluster)

    drawG2(g, node_size=1500, figsize=(20,10), label=True, edge=True, weight='sel', cluster=True)
    
