'''
Compare overlaps between top and frequent words
'''

import json
import os


dlist = [
    'amazon',
    'dianping',
    'economy',
    'vaccine',
    'yelp_hotel',
    'yelp_rest',
]


def overlap(dlist):
    results = {}
    for datan in dlist:
        if datan not in results:
            results[datan] = 0.0
        
        with open('./top/'+datan+'_mi.txt') as dfile:
            miw = set(json.load(dfile))

        with open('./top/'+datan+'_freq.txt') as dfile:
            freqw = set(json.load(dfile))

        results[datan] = len(miw.intersection(freqw))/(len(miw.union(freqw)))

    print(json.dumps(results, indent=4))

overlap(dlist)
