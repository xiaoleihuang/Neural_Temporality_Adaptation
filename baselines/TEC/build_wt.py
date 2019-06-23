'''
Build pre-trained fasttext embedding for the baseline
'''
from gensim.models import KeyedVectors
import pickle
import numpy as np
from scipy import sparse

data_list = [
    ('amazon', 'amazon_year'),
    ('dianping', 'dianping_year'),
    ('vaccine', 'vaccine_year'),
    ('yelp_hotel', 'yelp_hotel_year'),
    ('yelp_rest', 'yelp_rest_year'),
    ('google', 'economy_year'),
]


def create_wt_google(data_pair):
    print('Working on: '+data_pair[1])
    print('Loading word vectors: Google')
    vec_dir = '~/Documents/w2v/w2v/'
    word_vectors = KeyedVectors.load_word2vec_format(
        '/home/xiaolei/Documents/w2v/w2v/google.model', binary=True
    )

    print('Loading tokenizer......')
    tok = pickle.load(open('./toks/'+data_pair[1]+'.pkl', 'rb'))

    print('Creating Embedding Matrix...............')
    # first create matrix
    vec_len = 300
    embedding_matrix = sparse.lil_matrix((len(tok.word_index) + 1, vec_len))

    for pair in zip(word_vectors.wv.index2word, word_vectors.wv.syn0):
        # add the word if the word in the tokenizer
        if pair[0] in tok.word_index:
            embedding_matrix[tok.word_index[pair[0]]] = pair[1]

    # save the matrix to the dir
    np.save('./wt/'+data_pair[1]+'.npy', embedding_matrix.toarray())


def load_ft_vec(dpath):
    '''load fasttext with vec file'''
    wv = dict()
    with open(dpath) as dfile:
        dfile.readline()
        for line in dfile:
            line = line.split()
            wv[line[0]] = [float(item) for item in line[1:]]
    return wv


def create_wt_ft(data_pair, mode='cbow'):
    """Fasttext"""
    print('Working on: '+data_pair[1])
    print('Loading word vectors: '+data_pair[0])

    vec_dir = '/home/xiaolei/Documents/w2v/w2v/'
    word_vectors = load_ft_vec(
        vec_dir+data_pair[0]+'_' + mode + '.vec')

    print('Loading tokenizer......')
    tok = pickle.load(open('./toks/'+data_pair[1]+'.pkl', 'rb'))

    print('Creating Embedding Matrix...............')
    # first create matrix
    vec_len = 200
    embedding_matrix = sparse.lil_matrix((len(tok.word_index) + 1, vec_len))

    for pair in word_vectors.items():
        # add the word if the word in the tokenizer
        if pair[0] in tok.word_index:
            embedding_matrix[tok.word_index[pair[0]]] = pair[1]

    # save the matrix to the dir
    np.save('./wt/'+data_pair[1]+'.npy', embedding_matrix.toarray())


for data_pair in data_list:
    if 'google' in data_pair[0]:
        create_wt_google(data_pair)
    else:
        create_wt_ft(data_pair)

