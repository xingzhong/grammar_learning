# quick CYK parsing 
# optimum alignment sequence through dynamic programming 
# input:
#   1. rule sets
#   2. terminal liklihood 
# output:
#   optimum decoding alignment and structural tree
import math

class qcyk(object):
    MIN = -1e11
    def __init__(self):
        self.gamma = {}
        # gamma is the inside probabality 
        self.tau = {}
        # tau is for tree backtracking 
        self.rules = {}
        self.rules_lhs = {}
        self.length = 0
        
    def initTerminal(self, fin):
        # load temrinal likelihoods
        self.length = 0
        try:
            f = open(fin, 'r')
        except IOError:
            print "[Termnal] input raw string"
            for line in fin.split('\n'):
                line = line.strip()
                if len(line) > 1:
                    start, end, terminal, logP = line.split(',')
                    self.gamma[(int(start), int(end), terminal.strip())] = float(logP)
                    self.tau[(int(start), int(end), terminal.strip())] = (None, None, -1)
                    if int(end) > self.length:
                        self.length = int(end)
        else:
            print "input file name"
            with f:
                for line in f:
                    start, end, terminal, logP = line.split(',')
                    self.gamma[(int(start), int(end), terminal.strip())] = float(logP)
                    self.tau[(int(start), int(end), terminal.strip())] = (None, None, -1)
                    if int(end) > self.length:
                        self.length = int(end)


    def initRules(self, fin):
        try:
            f = open(fin, 'r')
        except IOError:
            print "[Rules] input raw string"
            for line in fin.split('\n'):
                line = line.strip()
                if len(line) > 1:
                    a, b, c, p = line.split(',')
                    a, b, c, p = a.strip(), b.strip(), c.strip(), math.log(float(p))
                    if len(c) < 1: c = None
                    self.rules[(a,b,c)] = p
                    if a in self.rules_lhs:
                        self.rules_lhs[a].append( (b, c, p) )
                    else:
                        self.rules_lhs[a] = [ (b,c,p) ]
        else:
            with f:
                for line in f:
                    a, b, c, p = line.split(',')
                    a, b, c, p = a.strip(), b.strip(), c.strip(), math.log(float(p))
                    if len(c) < 1: c = None
                    self.rules[(a,b,c)] = p
                    if a in self.rules_lhs:
                        self.rules_lhs[a].append( (b, c, p) )
                    else:
                        self.rules_lhs[a] = [ (b,c,p) ]

    def initGrammar(self, fin):
        with open(fin, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue
                lhs, rhs = line.split('->')
                lhs = lhs.strip()
                rhs = map(lambda x: x.strip(), rhs.split())
                rhs[-1] = math.log(float(rhs[-1][1:-1]))
                if len(rhs) > 2 :
                    a,b,c,p = lhs, rhs[0], rhs[1], rhs[2]
                else:
                    a,b,c,p = lhs, rhs[0], None, rhs[1]
                self.rules[(a,b,c)] = p
                if a in self.rules_lhs:
                    self.rules_lhs[a].append( (b, c, p) )
                else:
                    self.rules_lhs[a] = [ (b,c,p) ]
        print self.rules

    def parse(self):
        lik = self.getGamma(0, self.length, 'S')
        if lik > qcyk.MIN:
            self.tree = self.tree(0, self.length, 'S')
            self.lik = lik
            return lik, self.tree

    def tree(self, start, end, nt):
        root = (nt, start, end, [])
        y, z, k = self.tau[(start, end, nt)]
        if y:
            root[-1].append( self.tree(start, k, y) )
        if z:
            root[-1].append( self.tree(k+1, end, z) )
        return root

    def pretty_print_tree(self, root, space=0):
        s = "%s%s[%d:%d]\n"%("  "*space, root[0], root[1], root[2])
        for c in root[-1]:
            s += self.pretty_print_tree(c, space=space+1)
        return s

    def __repr__(self):
        s = 'logLike = %.4f\n'%self.lik
        return s + self.pretty_print_tree(self.tree)

    def leafs(self, root):
        if len(root[-1]) == 0:
            return [root[:-1]]
        else:
            res = []
            for c in root[-1]:
                res.extend(self.leafs(c))
            return res

    def getGamma(self, i, j, v):
        # v[i:k:j] -> y z 
        
        if (i, j, v) in self.gamma:
            return self.gamma[(i, j, v)]
        elif (i > j) or (v not in self.rules_lhs):
            return qcyk.MIN
        else:
            y0, z0, k0, lik0 = None,None,-1,-1e10
            #print self.rules_lhs[v]
            for (y, z, logP) in self.rules_lhs[v]:
                # print i, j, v, y, z
                if z and (i!=j):
                    for k in range(i, j):
                        lik = self.getGamma(i, k, y) + self.getGamma(k+1, j, z) + logP
                        if lik > lik0:
                            y0, z0, k0, lik0 = y,z,k,lik
                elif not z:
                    #import ipdb; ipdb.set_trace()
                    lik = self.getGamma(i, j, y) + logP
                    if lik > lik0:
                        y0, z0, k0, lik0 = y,None,j,lik


            if k0 > -1:
                self.gamma[(i, j, v)] = lik0
                self.tau[(i, j, v)] = (y0, z0, k0)
                
            else:
                self.gamma[(i, j, v)] = qcyk.MIN
                self.tau[(i, j, v)] = (y0, z0, k0)

            return self.gamma[(i, j, v)]
            raise AssertionError("No rules for nonterminal %s"%v)

if __name__ == '__main__':
    parser = qcyk()
    #parser.initGrammar("../test/grammar.gr")
    #parser.initTerminal("../test/qcyk_cal_liks.csv")
    parser.initTerminal("""
        0,0,Variable,0.0
        1,1,powOp,0.0
        2,2,Number, 0.0
        3,3,plus,0.0
        4,4,Number,0.0
        5,5,multiply,0.0
        6,6,Variable,0.0
        7,7,multiply,0.0
        8,8,Number,0.0""")
    #parser.initRules('../test/qcyk_cal_rules.csv')
    parser.initRules("""
        S, Number, , 0.143
        S, Variable, , 0.143
        S, Open, Exper_Close, 0.143
        S, Factor, PowOp_Primary, 0.143
        S, Term, MulOp_Factor, 0.143
        S, Expr, AddOp_Term, 0.143
        S, AddOp, Term, 0.143
        Expr, Number, , 0.143
        Expr, Variable, , 0.143
        Expr, Open, Exper_Close, 0.143
        Expr, Factor, PowOp_Primary, 0.143
        Expr, Term, MulOp_Factor, 0.143
        Expr, Expr, AddOp_Term, 0.143
        Expr, AddOp, Term, 0.143
        Term, Number, , 0.2
        Term, Variable, , 0.2
        Term, Open, Exper_Close, 0.2
        Term, Factor, PowOp_Primary, 0.2
        Term, Term, MulOp_Factor, 0.2
        Factor, Number, , 0.25
        Factor, Variable, , 0.25
        Factor, Open, Exper_Close, 0.25
        Factor, Factor, PowOp_Primary, 0.25
        Primary, Number, , 0.33 
        Primary, Variable, , 0.33 
        Primary, Open, Exper_Close, 0.33 
        AddOp, plus, , 0.5
        AddOp, minus, , 0.5
        MulOp, multiply, , 0.5
        MulOp, divide, , 0.5
        Expr_Close, Expr, Close, 1.0
        PowOp_Primary, PowOp, Primary, 1.0
        MulOp_Factor, MulOp, Factor, 1.0
        AddOp_Term, AddOp, Term, 1.0
        Open, openP, , 1.0
        Close, closeP, , 1.0
        PowOp, powOp, , 1.0
        """)
    tree = parser.parse()
    print parser
    
