import itertools 
class T(object):
  def __init__(self, symbol):
    self._symbol = symbol
    self._pos = (-1, -1)
  def setPos(self, pos):
    self._pos = pos
  def __repr__(self):
    if self._pos[0] == self._pos[1]:
      return "%s[%s]"%(self._symbol, self._pos[0])
    else:
      return "%s[%s:%s]"%(self._symbol, self._pos[0], self._pos[1])


class NT(object):
  def __init__(self, symbol):
    self._symbol = symbol
    self._pos = (-1, -1)
  def setPos(self, pos):
    self._pos = pos
  def __repr__(self):
    return "%s[%s:%s]"%(self._symbol, self._pos[0], self._pos[1])


class Productions(dict):
  def match(self, t1=None, nt=None, t2=None):
    #print "test t1=%s, nt=%s, t2=%s"%(t1,nt,t2)
    return filter(lambda p: self._match(p, t1, nt, t2), self.keys())

  def _match(self, p, t1, nt, t2):
    if nt :
      # match nt
      return self._matchAll(p,T(t1),NT(nt),T(t2))
    else:
      return self._matchAll(p,T(t1),NT('*'),T(t2))
    return False

  def _matchAll(self, p, t1, nt, t2):
    #print '\t',list( itertools.izip_longest(p[1], (t1, nt, t2)) )
    return all(map(lambda x: self.cmp(x[0], x[1]), itertools.izip_longest(p[1], (t1, nt, t2))))
  
  def cmp(self, t1, t2):
    if t1 == t2:
      return True
    if t1 and t2 and t1._symbol == t2._symbol:
      return True
    if t1 and t1._symbol == '*':
      return True
    if t2 and t2._symbol == '*':
      return True
    return False

  def __repr__(self):
    return "Productions\n"+"\n".join(map(self.reprProd, self.keys()))

  def reprProd(self, x):
    return "%s -> %s"%(x[0], x[1])

if __name__ == '__main__':
  production = Productions()
  production[(NT('S'), (NT(None), NT('A'), NT(None)))] = 0.33
  production[(NT('S'), (NT(None), NT('B'), NT(None)))] = 0.33
  production[(NT('S'), (NT(None), NT('C'), NT(None)))] = 0.33
  production[(NT('A'), (T('a'), NT('A'), T('a')))] = 0.6
  production[(NT('A'), (T('a'), NT('B'), T('a')))] = 0.3
  production[(NT('A'), (NT(None),))] = 0.1
  production[(NT('B'), (T('b'), NT('A'), NT(None)))] = 0.33
  production[(NT('B'), (T('b'), NT('B'), NT(None)))] = 0.33
  production[(NT('B'), (T('b'), NT('C'), NT(None)))] = 0.33
  production[(NT('B'), (NT(None),))] = 0.01
  production[(NT('C'), (NT(None), NT('B'), T('c')))] = 0.6
  production[(NT('C'), (NT(None), NT('C'), T('c')))] = 0.3
  production[(NT('C'), (NT(None),))] = 0.1

  print production.match(t1='a', t2='a')
  print production.match(t1='b', t2='a')
  print production.match(t1='b')
  print production.match(t2='c')
  print production.match(t1='a', t2='a', nt='A')
  print production.match(t1='b', nt='C')
  print production.match(t2='c', nt='C')
  print production.match(t1='a', t2='b', nt='A')
  print production.match(t1='b', nt='C', t2='a')
  print production.match(t2='c', nt='C')

  print production
