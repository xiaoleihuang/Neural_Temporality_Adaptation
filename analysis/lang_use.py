"""
Analysis and plot the language usage shift over time via the mutual information to select top features.
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


def lang_use(dname, topn=1000, mode='year'):
    '''
        dname: the data name
        topn: the number of top significant unigram and bigram features
        mode: domain option, seasonal (month) or non-seasonal (year)
    '''
    print('Working on: ', dname)
    # load data
    data = dict()
    results = dict()

    print('Loading data....')
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
                data[line[1]] = {'x':[], 'y':[]}
            data[line[1]]['x'].append(line[0])
            data[line[1]]['y'].append(line[2])

    # loop through each time domain
    print('Loop through each temporal domain')
    for key in data:
        # build vectorizer, obtain score for each feature
        vect = CountVectorizer(ngram_range=(1, 1), min_df=2, max_features=15000)
        x = vect.fit_transform(data[key]['x'])
        scores = mutual_info_classif(x, data[key]['y'])
        
        # rank and extract features
        top_indices = list(np.argsort(scores)[::-1][:topn])
        feas = vect.get_feature_names()
        results[key] = set([feas[idx] for idx in top_indices])

    # save the results
    pickle.dump(
        results,
        open('./lang_use/'+dname+'_'+mode+str(topn)+'.pkl', 'wb')
    )
    return results


def overlap(results, topn):
    '''Calculate overlaps between each time domain'''
    overlaps = dict()
    for key in sorted(results.keys()):
        if key not in overlaps:
            overlaps[key] = dict()

        for key1 in sorted(results.keys()):
            if key1 == key:
                overlaps[key][key1] = 1.0
            else:
                overlaps[key][key1] = len(
                    results[key].intersection(results[key1]))/topn
    return overlaps


def viz_use(df, ticks, title='default', outpath='./lang_use/overlap.pdf'):
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
    cmap = plt.get_cmap("RdBu_r")

    viz_plot = sns.heatmap(
        df, mask=mask, annot=True, cbar=False,  
        ax=ax, annot_kws={"size": 36}, cmap=cmap, 
        vmin=df.values.min(), fmt='.3f', center=center
    )
    plt.xticks([item+0.5 for item in range(len(ticks))], ticks, rotation=0, fontsize=25)
    plt.yticks([item+0.5 for item in range(len(ticks))], ticks, rotation=0, fontsize=25)
    plt.xlabel('Temporal Domain', fontsize=25)
    plt.ylabel('Temporal Domain', fontsize=25)
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
    for dname in data_list:
        if not os.path.exists(
            './lang_use/'+dname[0]+'_'+mode+str(topn)+'.pkl'
        ):
            results = lang_use(dname[0], topn, mode)
        else:
            results = pickle.load(
                open('./lang_use/'+dname[0]+'_'+mode+str(topn)+'.pkl',
                'rb'))
        overlaps = overlap(results, topn)
        test = OrderedDict(overlaps)
        keys = list(sorted(test.keys()))
        df = pd.DataFrame(test)
        df = df[keys]
#        viz_use(df, title=dname, outpath='./lang_use/'+dname+'.pdf')
        viz_use(
            df, ticks=dname[1], title=dname[2], 
            outpath='./lang_use/'+dname[0]+'.pdf'
        )
