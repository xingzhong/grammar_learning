from bs4 import BeautifulSoup
import pprint 

def load(file):
    soup = BeautifulSoup(open(file))
    data = []
    for obj in soup.find_all('object'):
        idnum = obj['id']
        for d in obj.find_all('data:bbox'):
            data.append( (d.attrs, idnum ))
    return data

if __name__ == '__main__':
    data = load('dataset/1-11200.xgtf')
    pprint.pprint(data[1:5])
	
