'''
This script is to build diachronic word embeddings from other sources
'''
import pickle
import os
import numpy as np
from scipy import sparse


def load_ft_vec(dpath):
    '''load fasttext with vec file'''
    wv = dict()
    with open(dpath) as dfile:
        dfile.readline()
        for line in dfile:
            line = line.split()
            wv[line[0]] = [float(item) for item in line[1:]]
    return wv


def build_dia(data_pair, dia, mode='cbow'):
    """Fasttext"""
    print('Working on: '+data_pair[1])
    print('Loading word vectors: '+data_pair[0])

    vec_dir = '/home/xiaolei/Documents/w2v/baselines/'
    vec_dir = vec_dir+dia+'/aligned/' + data_pair[0]+'/'
    flist = [item for item in os.listdir(vec_dir) if item.endswith('.vec')]

    print('Loading tokenizer......')
    tok = pickle.load(open('../domain_tokenizer/'+data_pair[1]+'.tkn', 'rb'))

    vec_len = 200

    for filep in flist:
        word_vectors = load_ft_vec(vec_dir+filep)
        dm = int(filep.split('.')[0])

        print('Creating Embedding Matrix...............')
        # first create matrix
        embedding_matrix = sparse.lil_matrix((len(tok[dm].word_index) + 1, vec_len))
        
        for pair in word_vectors.items():
            # add the word if the word in the tokenizer
            if pair[0]+str(dm) in tok[dm].word_index:
                embedding_matrix[tok[dm].word_index[pair[0]+str(dm)]] = pair[1]

        # save the matrix to the dir
        np.save(dia+'/'+data_pair[0]+'/'+str(dm)+'.npy', embedding_matrix.toarray())


dia_list = ['kim', 'kulkarni', 'hamilton']

data_list = [
    ('economy', 'economy_year', ['1', '2', '3', '4', '5', '6', 'general']),
    ('vaccine', 'vaccine_year', ['1', '2', '3', '4', 'general']),
    ('amazon', 'amazon_year', ['1', '2', '3', '4', '5', '6', 'general']),
    ('dianping', 'dianping_year', ['1', '2', '3', '4', 'general']),
    ('yelp_hotel', 'yelp_hotel_year', ['1', '2', '3', '4', 'general']),
    ('yelp_rest', 'yelp_rest_year', ['1', '2', '3', '4', 'general']),
]

for data_pair in data_list:
    for dia in dia_list:
        build_dia(data_pair, dia)

