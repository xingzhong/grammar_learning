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
            table[((x,y), z, '+')] += 1
            table[((y,z), x, '-')] += 1
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
            table[(x,y,'<')] += 1
    return table, symbols

def bestBC2(table, symbols, ecm, ecmC):
    # looking for `best` biclustering among symbol-to-symbol table.
    # FIXME better optimization searching method required
    # Now brute-force for demo
    bestScore, best = -np.inf, None
    _total = 2*2*len(symbols)*len(symbols)
    _ind = 0
    for i in [1, 2]:
        for j in [1, 2]:
            for r in itertools.combinations(symbols,i):
                for c in itertools.combinations(symbols,j):
                    _ind = _ind + 1
                    if _ind%len(symbols):
                        print "Searching one group %.4f"%(_ind/float(_total))
                    bc = BiCluster()
                    
                    bc.loadTable(table, r, c)
                    bc.loadEcm(ecm, ecmC)
                    bc.build()
                    temp = bc.logGain()
                    
                    if temp > bestScore:
                        bestScore = temp
                        best = bc
    return best


def bestBC(table, symbols, ecm, ecmC):
    if len(table) == 0:
        return None
    bestScore, best = -np.inf, None
    ds = dict(table.most_common(30))
    total = float(sum(ds.values()))
    items = ds.keys()
    probs = map(lambda x: float(x)/total, ds.values())
    
    candidates = []
    for _ in range(5):
        r, c, _ = items[np.random.choice(len(items), p=probs)]
        bc = BiCluster()
        bc.loadTable(table, [r], [c])
        bc.loadEcm(ecm, ecmC)
        bc.build()
        score = bc.logGain()
        if np.isinf(score):
            print "inf"
        else:    
            delta = 1.0
            while(delta > 0):
                bc_new = bc
                for new in np.random.permutation(list(symbols)):        
                          
                    bc_new_c = BiCluster().update(bc, table, ecm, col=new)        
                    if bc_new_c and bc_new_c.logGain() > bestScore:
                        bc_new = bc_new_c 
                        best = bc_new_c.logGain()
                        
                    bc_new_r = BiCluster().update(bc, table, ecm, row=new)  
                    if bc_new_r and bc_new_r.logGain() > bestScore:
                        bc_new = bc_new_r
                        best = bc_new_r.logGain()
                        
                delta = bc_new.logGain() - bc.logGain()
                bc = bc_new
            candidates.append( bc )
    bestScore, best = -np.inf, None
    for c in candidates:
        if c and c.logGain() > bestScore:
            best = c
    return best


def attach(bcs, n, t, ecm, g):
    new = Nonterminal(n)
    for bc in bcs:
        prod = bc.attach(new, t, ecm)
        if prod:
            g = addProd(g, [prod])
    return g

def addProdM(grammar, prod, nt):
    from multi import T
    if not grammar:
        grammar = WeightedGrammar(Nonterminal("START"), prod)
    else:
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
    import pdb; pdb.set_trace()
        
    return grammar
            
def addProd(grammar, prod, nt):

    if not grammar:
        grammar = WeightedGrammar(Nonterminal("START"), prod)

    else:
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
    def __init__(self, op='<', alpha=0.05):
        self._rows = []
        self._cols = []
        self._logGain = 0.0
        self._ecm = None
        self._table = None
        self._sum = 0.0
        self._nt = None
        self._prods = None
        self.alpha = alpha
        self._op = op
    
    @staticmethod
    def update(bc, table, ecm, col=None, row=None):
        newBc = deepcopy(bc)
        #import pdb ; pdb.set_trace()
        
        if col :
            if col in set(newBc._cols):
                return None
            newBc._cols.append(col)
            m,n = newBc._table.shape
            newc = np.zeros((m,1))
            for i,r in enumerate(newBc._rows):
                newc[i,0] = table[(r, col, newBc._op)]
            
            newBc._table = np.hstack((newBc._table, newc))
            
            ecmRows = [i for i in itertools.product(newBc._rows, [col])]
            newecmc = newBc._extract(ecm, ecmRows, newBc.ecmCols, newBc._op)
        
            newBc._ecm = np.vstack((newBc._ecm, newecmc))
            
        elif row :
            if row in set(newBc._rows):
                return None
            newBc._rows.append(row)
            m,n = newBc._table.shape
            newr = np.zeros((1,n))
            for i,c in enumerate(newBc._cols):
                newr[0,i] = table[(row, c, newBc._op)]
            newBc._table = np.vstack((newBc._table, newr))
            
            ecmRows = [i for i in itertools.product([row], newBc._cols )]
            newecmc = newBc._extract(ecm, ecmRows, newBc.ecmCols, newBc._op)
        
            newBc._ecm = np.vstack((newBc._ecm, newecmc))
        
        
        newBc.build()

        return newBc
        
    def loadTable(self, s2s, row, col):
        # given rows, cols, and symbol2symbol table
        # construct bicluster group
        self._rows = list(row)
        self._cols = list(col)
        self._table = self._extract(s2s, self._rows, self._cols, self._op)
        
    def loadEcm(self, ecm, cols):
        # given rows, cols, and symbol2symbol table
        # construct bicluster group
        self.ecmRows = [i for i in itertools.product(self._rows, self._cols)]
        self.ecmCols = cols
        self._ecm = self._extract(ecm, self.ecmRows, self.ecmCols, self._op)
   
    @staticmethod
    def _extract(t, row, col, op):
        # general extract submatrix routine
        m = len(row)
        n = len(col)
        table = np.zeros((m,n))
        for i,r in enumerate(row):
            for j,c in enumerate(col):
                table[i,j] = t[(r,c,op)]
        return table
    
    @staticmethod
    def _showT(t, r, c):
        s =  "\t"
        s += "\t".join(map(str, c))
        s += "\n"
        for i,x in enumerate(r):

            s += "%s\t"%str(x)
            for j,y in enumerate(c):
                s += "%s\t"%t[(i,j)]
            s += "\n"
        return s
    
    def __repr__(self):
        s = "BiCluster: %s (%s)\n"%(self._nt, self._op)
        s += "\tOR1 = %s \n"%(" | ".join(map(str, self._rows)))
        s += "\tOR2 = %s \n"%(" | ".join(map(str, self._cols)))
        s += "Symbol to Symbol Table:\n"
        s += self._showT(self._table, self._rows, self._cols)
        if DEBUG:
            s += "Expression Context Table:\n"
            s += self._showT(self._ecm, map(lambda x:" ".join(x), self.ecmRows), map(lambda x:"".join(x), self.ecmCols))
        s += "LogGain:\t %.4f = %.4f + %.4f +%.4f\n"%(
            self._logGain, self._logGain1, self._logGain2, self._logGain3)
        s += "Sum:\t\t %s\n"%self._sum
        if self._prods and DEBUG:
            s += "New Productions:\n"
            for p in self._prods:
                s += "%s\n"%p
        return s+'\n'
    
    def toRules(self):

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
        A,B = self._table.shape
        self._logGain3 = self.alpha * ( 4*self._sum - 2*(A+B) - 8 )
        
        self._logGain = self._logGain1 + self._logGain2 + self._logGain3
            
    
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
                sample[i:i+2] = [nt, '-1']
        return filter(lambda x: x != '-1', sample)
    
    def reduction(self, sample, nonTerminal=None):
        if nonTerminal :
            self._nt = Nonterminal(nonTerminal)
        new = map( lambda x: self._replace(x, self._nt, self._rows, self._cols), sample )
        return new

    def reductionM(self, samples, nonTerminal=None):
        if nonTerminal :
            self._nt = Nonterminal(nonTerminal)
        # do reduction on matrix column-wise and row-wise
        for i, sample in enumerate(samples):
            import pdb; pdb.set_trace()
            if self._op == '<':
                new = map( lambda x: self._replace(x, self._nt, self._rows, self._cols), sample )
                sample[i] = new[:]
                
            elif self._op == "=":
                pass
            
        #return sample
            

def learnGrammar(sample):
    sample2 = sample[:]
    g = None
    bcs = []
    totalBits = sum(map(len, sample2))
    print "total length %s \n"%totalBits
    for i in range(30):
        print "Compression :%.4f\n"%(sum(map(len, sample2))/float(totalBits))
        table, symbols = s2s(sample2)
        print "total symbols %s \n"%len(symbols)
        ecm, cols, _ = buildEcm(sample2)
        bc = bestBC(table, symbols, ecm, cols)
        if not bc: 
            print "no more rules!"
            break
        bcs.append(bc)
        new = 'NT_%s'%i
        sample2 = bc.reduction(sample2, new)
        prods = bc.toRules()
        g = addProd(g, prods, new)
        print "new"
        print bc
        table, symbols = s2s(sample2)
        ecm, cols, _ = buildEcm(sample2)

        for ind, _bc in enumerate(bcs):
            bc_new_c = BiCluster().update(_bc, table, ecm, col=Nonterminal(new))
            bc_new_r = BiCluster().update(_bc, table, ecm, row=Nonterminal(new))
            #print "bcG: %s"%bc_new.logGain()
            best = None
            if bc_new_c :
                bc_new = bc_new_c 
                best = bc_new_c.logGain()
            if bc_new_r and bc_new_r.logGain() > best:
                bc_new = bc_new_r
                best = bc_new_r.logGain()
            if best - bc.logGain() > 2.0:
                print "Attach"
                print bc_new
                g = addProd(g, bc_new.toRules(), bc_new._nt)
                sample2 = bc_new.reduction(sample2)
                bcs[ind] = bc_new
                
    g = postProcess(g, sample2)
    print "finish"
    return g, sample2

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
        if not bc: 
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
            if bc_new and bc_new.logGain() > 0.0:
                print "Adding col %s to %s"%(new, bc_new._nt)
                print bc_new
                g = addProd(g, bc_new.toRules(), bc_new._nt)
                sample2 = bc_new.reduction(sample2)
                bcs.append(bc_new)
                #continue
            
            bc_new = BiCluster().update(_bc, table, ecm, row=Nonterminal(new))
            #print "bcG: %s"%bc_new.logGain()
            if bc_new and bc_new.logGain() > 0.0:
                print "Adding col %s to %s"%(new, bc_new._nt)
                print bc_new
                g = addProd(g, bc_new.toRules(), bc_new._nt)
                sample2 = bc_new.reduction(sample2)
                bcs.append(bc_new)
                #continue
            
    g = postProcess(g, sample2)
    print "finish"


def s2sM(seqs):
    # generate symbol to symbol table 
    # input: (m,n) sequence of symbols. 
    #        m is the number of instance, 
    #        n is the number of the symbols in each instance
    # output: dict() key:tuple(x,y) value:occurrence 
    table = Counter()
    #symbols = reduce(lambda x,y: x.union(y), map(set, seqs), set())
    symbols = set()
    for seq in seqs:
        for s in seq:
            for x,y in zip(s, s[1:]):
                if x and y:
                    table[(x, y, '<')] += 1
                    symbols.add(x)
                    symbols.add(y)
                    #import pdb ; pdb.set_trace()
        for s in seq.T:
            for aid1, aid2 in itertools.combinations(range(len(s)), 2):
                if s[aid1] and s[aid2]:
                    table[(s[aid1], s[aid2], '=')] += 1   

    return table, symbols

def buildEcmM(seqs):
    # construct the expression context matrix
    table = Counter()
    rows = set()
    cols = set()
    for seq in seqs:
        for aid, s in enumerate(seq):
            for x,y,z in zip(s, s[1:], s[2:]):
                if x and y and z:
                    table[( (x, y), z, '<' )] += 1
                    table[( (y, z), x, '>')] += 1
                    rows.add((x, y))
                    rows.add((y, z))
                    cols.add((z, '<'))
                    cols.add((x, '>'))
        for s in seq.T:
            for aid1, aid2, aid3 in itertools.combinations(range(len(s)), 3):
                if s[aid1] and s[aid2] and s[aid3]:
                    table[( (s[aid1], s[aid2]), s[aid3], '=')] += 1
                    table[( (s[aid2], s[aid3]), s[aid1], '=')] += 1
                    rows.add((s[aid1], s[aid2]))
                    rows.add((s[aid2], s[aid3]))
                    cols.add((s[aid3], '='))
                    cols.add((s[aid1], '='))
                    
    return table, cols, rows

def preProcess(sample):
    for s in sample :
        for ind, ss in enumerate(s):
            ss[:] = map(lambda x: x and (x, ind), ss )
    return sample

def multi():
    import string
    from pprint import pprint
    sample = np.random.choice(['A','T','C','G', None], (50,3,10))
    sample = np.asarray(sample, dtype= np.dtype("object") )
    sample2 = sample.copy()
    bcs = []
    for i in range(1):
        sample2 = preProcess(sample2)
        table, symbols = s2sM(sample2)
        ecm, cols, rows = buildEcmM(sample2)
        bc = DupbestBC(table, symbols, ecm, cols)
        print bc
        if not bc: 
            print "no more !"
            break
        bcs.append(bc)
        new = 'NT_%s'%i
        sample2 = bc.reductionM(sample2, new)
        print sample2
    

def DupbestBC(table, symbols, ecm, ecmC, alpha=0.05, beta=5, cut=30):
    # alpha : trade-off parameter 
    # beta: 
    if len(table) == 0:
        return None
    bestScore, best = -np.inf, None
    ds = dict(table.most_common(cut))
    total = float(sum(ds.values()))
    items = ds.keys()
    probs = map(lambda x: float(x)/total, ds.values())
    #import pdb ; pdb.set_trace()
    candidates = []
    for _ in range(beta):
        r, c, op = items[np.random.choice(len(items), p=probs)]
        bc = BiCluster(op, alpha=alpha)
        bc.loadTable(table, [r], [c])
        bc.loadEcm(ecm, ecmC)
        bc.build()
        score = bc.logGain()
        if np.isinf(score):
            print "inf"
        else:
            symbolsList = list(symbols)
            delta = 1.0
            while(delta > 0):
                bc_new = bc
                for newIdx in np.random.permutation(len(symbolsList)):
                    new = symbolsList[newIdx]            
                    bc_new_c = BiCluster().update(bc, table, ecm, col=new)        
                    if bc_new_c and bc_new_c.logGain() > bestScore:
                        bc_new = bc_new_c 
                        best = bc_new_c.logGain()
                        
                    bc_new_r = BiCluster().update(bc, table, ecm, row=new)  
                    if bc_new_r and bc_new_r.logGain() > bestScore:
                        bc_new = bc_new_r
                        best = bc_new_r.logGain()
                        
                delta = bc_new.logGain() - bc.logGain()
                bc = bc_new
            candidates.append( bc )
    bestScore, best = -np.inf, None
    for c in candidates:
        if c and c.logGain() > bestScore:
            best = c
    return best

if __name__ == '__main__':
    #main()
    multi()