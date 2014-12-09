from qcyk import qcyk

# use sample data to test qcyk
# http://www.sussex.ac.uk/Users/johnca/cfg-resources/index.html



def convert_grammar():
	with open('ct-grammar-eval.txt', 'r') as f:
		res = f.read().split('\n\n')
	res = filter(lambda x: len(x)> 0 and (x[0]!=';'), res)
	for r in res[:50]:
		rs = r.split('\n')
		nt = rs[0]
		for x in rs[1:]:
			xs = x.split()
			if len(xs) > 2:
				print nt, xs
	
def main():
	parser = qcyk()

if __name__ == '__main__':
	convert_grammar()
	main()