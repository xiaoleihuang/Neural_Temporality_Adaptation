"""
1. This document tries to extract the top keywords in each domain by TF-IDF
2. Extract the top keywords by MI then extract their scores in the previous domain
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np


def rank_tfidf(data_name, top_num=100):
    data_path = '../raw_tsv_data/' + data_name + '.tsv'
    uniq_domains = set()
    domain_top_words = dict()
    
    # get domain information
    with open(data_path) as data_file:
        # skip the 1st column line
        data_file.readline()
        for line in data_file:
            infos = line.strip().split('\t')
            if len(infos) != 3:
                continue
            uniq_domains.add(infos[1])

    # loop the documents again
    for dkey in uniq_domains:
        domain_top_words[dkey] = dict()
        domain_docs = []

        with open(data_path) as data_file:
            # skip the 1st column line
            data_file.readline()
            for line in data_file:
                infos = line.strip().split('\t')
                if len(infos) != 3:
                    continue
                if infos[1] != dkey:
                    continue
                domain_docs.append(infos[0].strip())
        print(dkey)
        print(len(domain_docs))
        domain_vect = TfidfVectorizer(min_df=2, ngram_range=(1,3), stop_words='english')
        domain_docs = domain_vect.fit_transform(domain_docs)

        # select the top features
        tfidf_scores = np.asarray(domain_docs.mean(axis=0))[0]
        top_indices = list(reversed(tfidf_scores.argsort()))[:top_num]
        top_scores = dict()
        for top_idx in top_indices:
            top_scores[top_idx] = tfidf_scores[top_idx]

        # lookup dictionary of features
        for word_key in domain_vect.vocabulary_:
            if domain_vect.vocabulary_[word_key] in top_scores:
                domain_top_words[dkey][word_key] = top_scores[domain_vect.vocabulary_[word_key]]

    print(domain_top_words)
    with open('./tfidf/'+data_name+'.pkl', 'wb') as write_file:
        pickle.dump(domain_top_words, write_file)


def visualize_tfidf():
    pass
    # visualize
    #for dkey in sorted(uniq_domains):
    #    pass

data_list = [
        ('vaccine', 'vaccine_year'),
        ('amazon', 'amazon_month'),
        ('amazon', 'amazon_year'),
        ('dianping', 'dianping_month'),
        ('dianping', 'dianping_year'),
        ('google', 'economy_month'),
        ('google', 'economy_year'),
        ('google', 'parties_year'),
        ('vaccine', 'vaccine_month'),
        ('yelp_hotel', 'yelp_hotel_month'),
        ('yelp_hotel', 'yelp_hotel_year'),
        ('yelp_rest', 'yelp_rest_month'),
        ('yelp_rest', 'yelp_rest_year'),
    ]

for pair in data_list:
    rank_tfidf(pair[1])
