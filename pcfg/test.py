import unittest
from production import *

class test(unittest.TestCase):
	def setUp(self):
		self.production = Productions()
		self.production[(NT('S'), (None, NT('A'), None))] = 0.33
		self.production[(NT('S'), (None, NT('B'), None))] = 0.33
		self.production[(NT('S'), (None, NT('C'), None))] = 0.33
		self.production[(NT('A'), (T('a'), NT('A'), T('a')))] = 0.6
		self.production[(NT('A'), (T('a'), NT('B'), T('a')))] = 0.3
		self.production[(NT('A'), (None,))] = 0.1
		self.production[(NT('B'), (T('b'), NT('A'), None))] = 0.33
		self.production[(NT('B'), (T('b'), NT('B'), None))] = 0.33
		self.production[(NT('B'), (T('b'), NT('C'), None))] = 0.33
		self.production[(NT('B'), (None,))] = 0.01
		self.production[(NT('C'), (None, NT('B'), T('c')))] = 0.6
		self.production[(NT('C'), (None, NT('C'), T('c')))] = 0.3
		self.production[(NT('C'), (None,))] = 0.1


if __name__ == '__main__':
    unittest.main()