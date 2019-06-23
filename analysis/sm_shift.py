"""
Analysis and plot the word semantic meaning shift over time via wasserstein_distance
"""
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance, ttest_ind

import os
import json
from collections import OrderedDict
import pickle


def get_domain(dname, mode='year'):
    '''Obtain the domain list
    '''
    # load data
    domains = set()

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

            domains.add(int(line[1]))
    domains = sorted(list(domains))
    with open('./sm_shift/domain/'+dname+'_'+mode+'.txt', 'w') as wfile:
        wfile.write(json.dumps(domains, ensure_ascii=False))
    return domains


def freq_words(dname, topn=1000, mode='year', reverse=False):
    '''Find the most frequent words
    '''
    # load data
    counts = dict()
    if reverse:
        reverse_tag = '_re'
    else:
        reverse_tag = ''

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

            line = [word.strip() for word in line[0].split() if len(word)>1]
            for word in line:
                if word not in counts:
                    counts[word] = 0
                counts[word] += 1
    # rank and select the top frequent words
    sorts = sorted(counts.items(), key=lambda kv:kv[1], reverse=reverse)[:topn]
    sorts = [item[0] for item in sorts]
    
    # save and return
    with open('./sm_shift/top/'+dname+'_freq'+reverse_tag+'.txt', 'w') as wfile:
        wfile.write(json.dumps(sorts, ensure_ascii=False))
    return sorts


def mutual_words(dname, topn=1000, mode='year', reverse=False):
    '''Find the most important words by mutual information
    '''
    # load data
    data = {'x':[], 'y':[]}
    if reverse:
        reverse_tag = '_re'
    else:
        reverse_tag = ''

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

            line[2] = int(line[2])
            data['x'].append(line[0])
            data['y'].append(line[2])

    print('Vectorizing.....')
    vect = TfidfVectorizer(ngram_range=(1, 1), min_df=2, max_features=15000)
    x = vect.fit_transform(data['x'])
    print('Mutual Information........')
    scores = mutual_info_classif(x, data['y'])
    # rank and extract features
    top_indices = list(np.argsort(scores)[::-1])
    if reverse:
        top_indices = top_indices[-topn:]
    else:
        top_indices = top_indices[:topn]

    feas = vect.get_feature_names()
    results = [feas[idx] for idx in top_indices]

    # save and return
    with open('./sm_shift/top/'+dname+'_mi'+reverse_tag+'.txt', 'w') as wfile:
        wfile.write(json.dumps(results, ensure_ascii=False))
    return results


def extract_vec(dname, words, domains, suffix='mi', mode='year', reverse=False):
    '''Extract the vectors of the words across different domains
        
    '''
    results = dict()
    if reverse:
        reverse_tag = '_re'
    else:
        reverse_tag = ''

    for domain in domains:
        # word dictionary
        wdic = dict([(item+str(domain), idx) for idx, item in enumerate(words)])

        # load the pre-trained embedding
        vec_path = '/home/xiaolei/Documents/w2v/fasttext_cbow/'+ \
            dname+'/'+mode+'/'+dname+'.vec'

        with open(vec_path) as dfile:
            # dim: vector dimension
            dim = int(dfile.readline().strip().split()[1])
            wmatrix = [[0]*dim] * len(words)

            for line in dfile:
                line = line.strip().split()
                if line[0] in wdic:
                    wmatrix[wdic[line[0]]] = [float(item) for item in line[1:]]

        wmatrix = np.mean(np.asarray(wmatrix), axis=0)
        results[domain] = wmatrix
    # save and return
    pickle.dump(results, open(
        './sm_shift/matrix/'+dname+'_'+mode+'_'+suffix+reverse_tag+'.pkl',
        'wb')
    )
    return results


def calc_wd(results, dname, suffix='mi', reverse=False):
    '''Compare the distribution differences across domains via wasserstein distance
    
        suffix: two options, mutual information (mi) and frequence (freq)
    '''
    dist = dict()
    for idm in results:
        if idm not in dist:
            dist[idm] = dict()
        for jdm in results:
            if idm == jdm:
                dist[idm][jdm] = 0.0
            dist[idm][jdm] = wasserstein_distance(
                results[idm], results[jdm]
            )
    
    # save and return
    print(dist)
    with open('./sm_shift/wd/'+dname+'_'+suffix+reverse_tag+'.json', 'w') as wfile:
        wfile.write(json.dumps(dist, ensure_ascii=False))
    return dist


def viz_wd(df, ticks, title='default', outpath='./sm_shift/wd.pdf'):
    """
        Heatmap visualization for wasstertain distance
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

    center = np.median([item for item in df.to_numpy().ravel() if item != 1])
    
    a4_dims = (16.7, 12.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.set(font_scale=1.2)
    viz_plot = sns.heatmap(
        df, mask=mask, annot=True, cbar=False,  
        ax=ax, annot_kws={"size": 36}, cmap='RdBu_r', 
        vmin=df.values.min(), fmt='.3f', center=center
    ) # center=0,
    plt.xticks([item+0.5 for item in range(len(ticks))], ticks, rotation=0, fontsize=25)
    plt.yticks([item+0.5 for item in range(len(ticks))], ticks, rotation=0, fontsize=25)
    plt.xlabel('Temporal Domain', fontsize=25)
    plt.ylabel('Temporal Domain', fontsize=25)
    plt.title(title, fontsize=36)
    ax.set_facecolor("white")
    viz_plot.get_figure().savefig(outpath, format='pdf')
    plt.close()


def sig_analysis():
    '''Significance analysis on the Yelp-hotel case'''
    base = [0]*6
    top = [.227, .326, .225, .117, .164,.071]
    freq = [.001, .001, .001, .0, .0, .0]

    print(
        'Frequent: ', 
        stats.ttest_ind(freq, base, equal_var=False)
    )
    
    print(
        'Top: ', 
        stats.ttest_ind(top, base, equal_var=False)
    )


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
    suffix='freq' # two different modes: mi or freq
    reverse=False

    if reverse:
        reverse_tag = '_re'
    else:
        reverse_tag = ''

    for dname in data_list:
        print('Working on: ', dname[0])

        # obtain the data temporal domains
        print('Obtain the domain list......')
        if not os.path.exists('./sm_shift/domain/'+dname[0]+'_'+mode+'.txt'):
            domains = get_domain(dname[0], mode=mode)
        else:
            domains = json.load(open('./sm_shift/domain/'+dname[0]+'_'+mode+'.txt'))

        '''Mutual Information Or Frequency'''
        print('Obtain the top features via', suffix)
        # rank the top words/features
        if not os.path.exists('./sm_shift/top/'+dname[0]+'_'+suffix+reverse_tag+'.txt'):
            if suffix == 'mi':
                top_words = mutual_words(dname[0], topn=topn, mode=mode, reverse=reverse)
            elif suffix == 'freq':
                top_words = freq_words(dname[0], topn=topn, mode=mode, reverse=reverse)
        else:
            with open('./sm_shift/top/'+dname[0]+'_'+suffix+reverse_tag+'.txt') as dfile:
                top_words = json.load(dfile)

        # extract vectors for mutual information
        print('Obtain domain distributions......')
        if not os.path.exists('./sm_shift/matrix/'+dname[0]+'_'+mode+'_'+suffix+reverse_tag+'.pkl'):
            dist_vec = extract_vec(
                dname[0], top_words, domains, 
                suffix=suffix, mode=mode, reverse=reverse
            )
        else:
            dist_vec = pickle.load(
                open('./sm_shift/matrix/'+dname[0]+'_'+mode+'_'+suffix+reverse_tag+'.pkl', 'rb')
            )

        # calculate the distanaces between domains
        print('Calculating cross domain distances')
        if not os.path.exists('./sm_shift/wd/'+dname[0]+'_'+suffix+reverse_tag+'.json'):
            diffs = calc_wd(dist_vec, dname[0], suffix=suffix, reverse=reverse)
        else:
            with open('./sm_shift/wd/'+dname[0]+'_'+suffix+reverse_tag+'.json') as dfile:
                diffs = json.load(dfile)

        # visualization
        diffs = OrderedDict(diffs)
        df = pd.DataFrame(diffs)
        try:
            df = df[domains]
        except:
            df = df[map(str,domains)]
        viz_wd(
            df, ticks=dname[1], title=dname[2], 
            outpath='./sm_shift/plot'+reverse_tag+'/'+dname[0]+'_'+ \
                mode+'_'+suffix+reverse_tag+'.pdf'
        )

