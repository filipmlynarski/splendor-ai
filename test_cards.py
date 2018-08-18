import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import pandas as pd

#valiadte csv file
df = pd.read_csv('cards.csv')
print(df.ix[0:4])