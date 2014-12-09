import unittest
from qcyk import qcyk

class qcky_test(unittest.TestCase):
	def setUp(self):
		self.qcyk = qcyk()
	def test(self):
		self.assertEqual(1+1,2)
	def test_load_terminal(self):
		self.qcyk.initTerminal("qcyk_test.csv")
		n = len(self.qcyk.gamma)
		num_lines = sum(1 for line in open("qcyk_test.csv"))
		self.assertEqual(n, num_lines)
	def test_load_rules(self):
		self.qcyk.initRules('qcyk_test_rules.csv')

if __name__ == '__main__':
	unittest.main()