"""
Analysis and plot the word contexts shift over time
"""
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os, sys
from collections import OrderedDict


def ctt_shift(dname, topn=1000, window=5, mode='year'):
    '''
        dname: the data name
        topn: the number of top significant unigram features
        window: context window
        mode: domain option, seasonal (month) or non-seasonal (year)
    '''
    print('Working on: ', dname)
    data = {'x':[], 'y':[]}

    # load the data to get top features
    print('Getting top words....')
    if not os.path.exists('./ctt_shift/'+dname+'.tws'):
        with open('../raw_tsv_data/'+dname+'_'+mode+'.tsv') as dfile:
            dfile.readline() # skip the header
            for line in dfile:
                line = line.strip()
                if len(line) < 2:
                    continue

                line = line.split('\t')
                if len(line) != 3:
                    continue

                line[1] = int(line[1])
                line[2] = int(line[2])
                
                data['x'].append(line[0])
                data['y'].append(line[2])

        # vectorize and extract the top 1000 features
        vect = CountVectorizer(ngram_range=(1, 1), max_df=int(len(data['x'])/2), min_df=2, max_features=15000)
        x = vect.fit_transform(data['x'])
        scores = mutual_info_classif(x, data['y'])

        # release memory
        del x
        del data

        # rank and extract features
        top_indices = list(np.argsort(scores)[::-1][:topn])
        feas = vect.get_feature_names()
        twords = set([feas[idx] for idx in top_indices])

        # release memory
        del feas
        del scores
        del vect

        # save the twords
        pickle.dump(twords, open('./ctt_shift/'+dname+'.tws', 'wb'))
    else:
        twords = pickle.load(open('./ctt_shift/'+dname+'.tws', 'rb'))

    # load data again to obtain domain specific data
    print('Loading data again')
    data = dict()
    with open('../raw_tsv_data/'+dname+'_'+mode+'.tsv') as dfile:
        dfile.readline() # skip the header
        for line in dfile:
            line = line.strip()
            if len(line) < 2:
                continue

            line = line.split('\t')
            if len(line) != 3:
                continue

            line[1] = int(line[1])
            line[2] = int(line[2])
            
            if line[1] not in data:
                data[line[1]] = []
            
            data[line[1]].append([item for item in line[0].split() if len(item)>0])

    # loop through each temporal domain to find contexts of top features
    print('Counting contextual words')
    ctts = dict()
    for key in data:
        if key not in ctts:
            ctts[key] = set()

        # find the contextual words
        for line in data[key]:
            tmp_dic = dict()
            # record the word and its idx
            for idx, word in enumerate(line):
                if word not in tmp_dic:
                    tmp_dic[word] = []
                tmp_dic[word].append(idx)

            # loop through each feature
            for tw in twords:
                if tw not in tmp_dic:
                    continue
                # collect contextual words around the index
                for idx in tmp_dic[tw]:
                    start = idx - window
                    if start < 0:
                        start = 0

                    # +1 because the slice ends 1 earlier
                    end = idx + window + 1 
                    if end > len(line):
                        end = len(line)

                    # left and right contexts
                    ctts[key].update(line[start:idx])
                    ctts[key].update(line[idx:end])

    # save and return the ctts
    pickle.dump(
        ctts, open('./ctt_shift/'+dname+'.ctt', 'wb')
    )
    return ctts


def overlap(ctts, topn):
    '''Calculate contextual words overlaps between each time domain'''
    overlaps = dict()
    for key in ctts:
        if key not in overlaps:
            overlaps[key] = dict()

        for key1 in ctts:
            if key1 == key:
                overlaps[key][key1] = 1.0
            else:
                overlaps[key][key1] = len(
                    ctts[key].intersection(ctts[key1]))/len(
                        ctts[key].union(ctts[key1])
                    )
    return overlaps


def viz_use(df, ticks, title='default', outpath='./ctt_shift/overlap.pdf'):
    """
        Heatmap visualization
        :param df: an instance of pandas DataFrame
        :return: 
    """
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask, 1)] = True
    for idx in range(len(mask)):
        for jdx in range(len(mask[0])):
            if idx == jdx:
                mask[idx][jdx] = False
#    print(mask)
    center = np.median([item for item in df.to_numpy().ravel() if item != 1])

    a4_dims = (16.7, 12.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.set(font_scale=1.2)
    viz_plot = sns.heatmap(
        df, mask=mask, annot=True, cbar=False,  
        ax=ax, annot_kws={"size": 36}, cmap='RdBu_r', 
        vmin=df.values.min(), fmt='.3f', center=center
    ) # center=0,
    plt.xlabel('Temporal Domain', fontsize=25)
    plt.ylabel('Temporal Domain', fontsize=25)
    plt.xticks([item+0.5 for item in range(len(ticks))], ticks, rotation=0, fontsize=25)
    plt.yticks([item+0.5 for item in range(len(ticks))], ticks, rotation=0, fontsize=25)
    plt.title(title, fontsize=36)
    ax.set_facecolor("white")
    viz_plot.get_figure().savefig(outpath, format='pdf')
    plt.close()


if __name__ == '__main__':
    data_list = [
        ['vaccine', ['2013', '2014', '2015', '2016'], 'Twitter'],
        [
            'amazon', 
            ['1997-99', '2000-02', '2003-05', '2006-08', '2009-11', '2012-14'], 
            'Amazon'
        ],
        ['yelp_rest', ['2006-08', '2009-11', '2012-14', '2015-17'], 'Yelp-rest'],
        ['yelp_hotel', ['2006-08', '2009-11', '2012-14', '2015-17'], 'Yelp-hotel'],
        ['dianping', ['2009', '2010', '2011', '2012'], 'Dianping'],
        ['economy', ['1950-70', '1971-85', '1986-2000', '2001-14'], 'Economy'],
    ]

    topn = 1000
    mode = 'year'
    window = 5
    for dname in data_list:
        if not os.path.exists('./ctt_shift/'+dname[0]+'.ctt'):
            results = ctt_shift(dname[0], topn, window, mode)
        else:
            results = pickle.load(open('./ctt_shift/'+dname[0]+'.ctt', 'rb'))
        overlaps = overlap(results, topn)
        test = OrderedDict(overlaps)
        keys = list(sorted(test.keys()))
        df = pd.DataFrame(test)
        df = df[keys]
        viz_use(
            df, ticks=dname[1], title=dname[2], 
            outpath='./ctt_shift/'+dname[0]+'.pdf'
        )
