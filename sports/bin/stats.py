import glob
import pandas as pd
import matplotlib.pyplot as plt

def main():
  data = []
  for idx, fn in enumerate(glob.glob("../newData/*_2/decode.csv")):
    print fn
    df = pd.read_csv(fn, header=None, index_col=False, names=['NT', 'time', 'depth', 'logLik'])
    df['file'] = idx
    data.append(df)
  data = pd.concat(data)
  print data.groupby(['NT', 'file']).count()
  #import pdb; pdb.set_trace()
  print len(data)
  data[['NT','file']].hist(by='NT', bins=len(data.file.unique()), color='c')
  plt.title("total frames # %s"%len(data))
  plt.savefig('stats.eps', dps=200)
  

if __name__ == '__main__':
  main()
