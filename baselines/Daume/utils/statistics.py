from collections import Counter
import pandas as pd

def domain_dist(datapath):
    test = pd.read_csv(datapath)
    print(Counter(test['Time']))