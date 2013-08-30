from collections import Counter, OrderedDict
from nltk.grammar import WeightedGrammar, WeightedProduction, Nonterminal, induce_pcfg
from nltk.parse import pchart
from copy import deepcopy
import numpy as np
import itertools

DEBUG = False

def buildEcm(seqs):
    # construct the expression context matrix
    table = Counter()
    rows = set()
    cols = set()
    for seq in seqs:
        for x,y,z in zip(seq, seq[1:], seq[2:]):
            table[((x,y), (z, '+'))] += 1
            table[((y,z), (x, '-'))] += 1
            rows.add((x,y))
            rows.add((y,z))
            cols.add((z, '+'))
            cols.add((x, '-'))
    return table, cols, rows

def s2s(seqs):
    # generate symbol to symbol table 
    # input: (m,n) sequence of symbols. 
    #        m is the number of instance, 
    #        n is the number of the symbols in each instance
    # output: dict() key:tuple(x,y) value:occurrence 
    table = Counter()
    symbols = reduce(lambda x,y: x.union(y), map(set, seqs), set())
    for seq in seqs:
        for x,y in zip(seq, seq[1:]):
            table[(x,y)] += 1
    return table, symbols

def bestBC(table, symbols, ecm, ecmC):
    # looking for `best` biclustering among symbol-to-symbol table.
    # FIXME better optimization searching method required
    # Now brute-force for demo
    bestScore, best = -np.inf, None
    for i in [1, 2]:
        for j in [1, 2]:
            if i == 1 and j == 1:
                continue
            else:
                for r in itertools.combinations(symbols,2):
                    for c in itertools.combinations(symbols,1):
                        bc = BiCluster()
                        
                        bc.loadTable(table, r, c)
                        bc.loadEcm(ecm, ecmC)
                        bc.build()
                        temp = bc.logGain()
                        
                        if temp > bestScore:
                            bestScore = temp
                            best = bc
    return best

def attach(bcs, n, t, ecm, g):
    new = Nonterminal(n)
    for bc in bcs:
        prod = bc.attach(new, t, ecm)
        if prod:
            g = addProd(g, [prod])
    return g
            
def addProd(grammar, prod, nt):

    if not grammar:
        grammar = WeightedGrammar(Nonterminal("START"), prod)

    else:
        print nt
        #prods = filter(lambda x: not str(x.lhs()).startswith(str(nt)), grammar._productions)
        prods = []
        for p in grammar._productions:
            x = str(p.lhs())
            test = set([str(nt), "%s_A"%str(nt), "%s_B"%str(nt)])
            if x in test:
                continue
            else:
                prods.append(p)
        prods.extend(prod)

        grammar = WeightedGrammar(Nonterminal("START"), prods)
        
    return grammar
    
def postProcess(g, sample):
    final = Counter()
    total = 0
    for s in sample:
        if len(s) < 2:
            final[s[0]] += 1
            total += 1
    prods = []
    for k,v in final.iteritems():
        prods.append( WeightedProduction(Nonterminal("START"), [k], prob= v/float(total)) )
    g = addProd(g, prods, Nonterminal("START"))
    return g

def _xlogx(x):
    if x==0.0: return 0.0
    else:
        return x*np.log(x)

xlogx = np.frompyfunc(_xlogx, 1, 1)

class BiCluster():
    def __init__(self):
        self._rows = []
        self._cols = []
        self._logGain = 0.0
        self._ecm = None
        self._table = None
        self._sum = 0.0
        self._nt = None
        self._prods = None
    
    @staticmethod
    def update(bc, table, ecm, col=None, row=None):
        newBc = deepcopy(bc)
        if col:
            if col in set(newBc._cols):
                return None
            newBc._cols.append(col)
            m,n = newBc._table.shape
            newc = np.zeros((m,1))
            for i,r in enumerate(newBc._rows):
                newc[i,0] = table[(r, col)]
                
            newBc._table = np.hstack((newBc._table, newc))
            
            ecmRows = [i for i in itertools.product(newBc._rows, [col])]
            newecmc = newBc._extract(ecm, ecmRows, newBc.ecmCols)
        
            newBc._ecm = np.vstack((newBc._ecm, newecmc))
            
        elif row:
            if row in set(newBc._rows):
                return None
            newBc._rows.append(row)
            m,n = newBc._table.shape
            newr = np.zeros((1,n))
            for i,c in enumerate(newBc._cols):
                newr[0,i] = table[(row, c)]
            newBc._table = np.vstack((newBc._table, newr))
            
            ecmRows = [i for i in itertools.product([row], newBc._cols )]
            newecmc = newBc._extract(ecm, ecmRows, newBc.ecmCols)
        
            newBc._ecm = np.vstack((newBc._ecm, newecmc))
        
        
        newBc.build()

        return newBc
        
    def loadTable(self, s2s, row, col):
        # given rows, cols, and symbol2symbol table
        # construct bicluster group
        self._rows = list(row)
        self._cols = list(col)
        self._table = self._extract(s2s, self._rows, self._cols)
        
    def loadEcm(self, ecm, cols):
        # given rows, cols, and symbol2symbol table
        # construct bicluster group
        self.ecmRows = [i for i in itertools.product(self._rows, self._cols)]
        self.ecmCols = cols
        self._ecm = self._extract(ecm, self.ecmRows, self.ecmCols)
   
    @staticmethod
    def _extract(t, row, col):
        # general extract submatrix routine
        m = len(row)
        n = len(col)
        table = np.zeros((m,n))
        for i,r in enumerate(row):
            for j,c in enumerate(col):
                table[i,j] = t[(r,c)]
        return table
    
    @staticmethod
    def _showT(t, r, c):
        s =  "\t"
        s += "\t".join(map(str, c))
        s += "\n"
        for i,x in enumerate(r):
            s += "%s\t"%x
            for j,y in enumerate(c):
                s += "%s\t"%t[(i,j)]
            s += "\n"
        return s
    
    def __repr__(self):
        s = "BiCluster: %s\n"%self._nt
        s += "\tOR1 = %s \n"%(" | ".join(map(str, self._rows)))
        s += "\tOR2 = %s \n"%(" | ".join(map(str, self._cols)))
        s += "Symbol to Symbol Table:\n"
        s += self._showT(self._table, self._rows, self._cols)
        if DEBUG:
            s += "Expression Context Table:\n"
            s += self._showT(self._ecm, map(lambda x:" ".join(x), self.ecmRows), map(lambda x:"".join(x), self.ecmCols))
        s += "LogGain:\t %s = %s + %s\n"%(self._logGain, self._logGain1, self._logGain2)
        s += "Sum:\t\t %s\n"%self._sum
        if self._prods and DEBUG:
            s += "New Productions:\n"
            for p in self._prods:
                s += "%s\n"%p
        return s+'\n'
    
    def toRules(self, nonTerminal=None):
        if nonTerminal:
            self._nt = Nonterminal(nonTerminal)
        else:
            nonTerminal = str(self._nt)
            
        prods = []
        col = np.sum(self._table, axis=0)
        row = np.sum(self._table, axis=1)
        A = Nonterminal(nonTerminal+'_A')
        B = Nonterminal(nonTerminal+'_B')
        C = Nonterminal(nonTerminal)
        for i, r in enumerate(self._rows):
            prods.append( WeightedProduction(A, [r], prob=float(row[i]/self._sum)) )
        for i, c in enumerate(self._cols): 
            prods.append( WeightedProduction(B, [c], prob=float(col[i]/self._sum)) )
        prods.append( WeightedProduction(C, [A,B], prob=1.0) )
        self._prods = prods
        return prods
        
    
    def build(self):
        if not np.all(self._table) : 
            self._logGain1 = -np.inf
        else:
            self._logGain1 = self._logM(self._table) 
        self._sum = np.sum(self._table)
        self._logGain2 = self._logM(self._ecm)

        self._logGain = self._logGain1 + self._logGain2 + self._sum/4
    
    @staticmethod
    def _logM(matrix):
        col = np.sum(matrix, axis=0)
        row = np.sum(matrix, axis=1)
        total = np.sum(matrix)
        
        return sum(xlogx(row)) + sum(xlogx(col)) - xlogx(total) - np.sum(xlogx(matrix))
    
    def logGain(self):
        return self._logGain
    
    @staticmethod
    def _replace(sample, nt, r, c):
        for i in range(len(sample) - 1):
            if sample[i] in r and sample[i+1] in c:
                sample[i:i+2] = [nt, None]
        return filter(lambda x:x, sample)
    
    def reduction(self, sample, nonTerminal=None):
        if nonTerminal :
            self._nt = Nonterminal(nonTerminal)
        new = map( lambda x: self._replace(x, self._nt, self._rows, self._cols), sample )
        if DEBUG:
            for n in new :
                print n
        return new

def main():
    import string
    sample = np.random.choice(list(string.lowercase[:4]), (50,10))
    sample = np.asarray(sample, dtype= np.dtype("object") )
    sample2 = sample.copy().tolist()
    g = None
    bcs = []
    totalBits = sum(map(len, sample2))
    for i in range(30):
        print "alpha:%s\n"%(sum(map(len, sample2))/float(totalBits))
        table, symbols = s2s(sample2)
        ecm, cols, _ = buildEcm(sample2)
        bc = bestBC(table, symbols, ecm, cols)
        if np.isneginf(bc.logGain()): 
            print "no more !"
            break
        bcs.append(bc)
        new = 'NT_%s'%i
        sample2 = bc.reduction(sample2, new)
        prods = bc.toRules(new)
        g = addProd(g, prods, new)
        print bc
        table, symbols = s2s(sample2)
        ecm, cols, _ = buildEcm(sample2)
        for _bc in bcs:
            bc_new = BiCluster().update(_bc, table, ecm, col=Nonterminal(new))
            #print "bcG: %s"%bc_new.logGain()
            if bc_new and bc_new.logGain() > 10.0:
                print "Adding col %s to %s"%(new, bc_new._nt)
                print bc_new
                g = addProd(g, bc_new.toRules(), bc_new._nt)
                sample2 = bc_new.reduction(sample2)
                bcs.append(bc_new)
                continue
            
            bc_new = BiCluster().update(_bc, table, ecm, row=Nonterminal(new))
            #print "bcG: %s"%bc_new.logGain()
            if bc_new and bc_new.logGain() > 10.0:
                print "Adding col %s to %s"%(new, bc_new._nt)
                print bc_new
                g = addProd(g, bc_new.toRules(), bc_new._nt)
                sample2 = bc_new.reduction(sample2)
                bcs.append(bc_new)
                continue
            
    g = postProcess(g, sample2)
    print "finish"

if __name__ == '__main__':
    main()
    