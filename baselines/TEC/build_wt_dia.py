'''
Build pre-trained fasttext embedding for the baseline
'''
from gensim.models import KeyedVectors
import pickle
import numpy as np
from scipy import sparse
import os

data_list = [
    ('amazon', 'amazon_year', ['1', '2', '3', '4', '5', '6']),
    ('dianping', 'dianping_year', ['1', '2', '3', '4']),
    ('vaccine', 'vaccine_year', ['1', '2', '3', '4']),
    ('yelp_hotel', 'yelp_hotel_year', ['1', '2', '3', '4']),
    ('yelp_rest', 'yelp_rest_year', ['1', '2', '3', '4']),
    ('economy', 'economy_year', ['1', '2', '3', '4', '5', '6']),
]

dia_list = ['kim', 'kulkarni', 'hamilton']


def load_ft_vec(dpath):
    '''load fasttext with vec file'''
    wv = dict()
    with open(dpath) as dfile:
        dfile.readline()
        for line in dfile:
            line = line.split()
            wv[line[0]] = [float(item) for item in line[1:]]
    return wv


def create_wt_ft(data_pair, dia, mode='cbow'):
    """Fasttext"""
    print('Working on: '+data_pair[1])
    print('Loading word vectors: '+data_pair[0])

    vec_dir = '/home/xiaolei/Documents/w2v/baselines/'
    vec_dir = vec_dir+dia+'/aligned/' + data_pair[0]+'/'
    flist = [item for item in os.listdir(vec_dir) if item.endswith('.vec')]

    print('Loading tokenizer......')
    tok = pickle.load(open('./toks/'+data_pair[1]+'.pkl', 'rb'))

    if data_pair[0] in ['vaccine', 'economy']:
        vec_len = 300
    else:
        vec_len = 200

    for filep in flist:
        word_vectors = load_ft_vec(vec_dir+filep)
        print('Creating Embedding Matrix...............')
        # first create matrix
        embedding_matrix = sparse.lil_matrix((len(tok.word_index) + 1, vec_len))

        for pair in word_vectors.items():
            # add the word if the word in the tokenizer
            if pair[0] in tok.word_index:
                embedding_matrix[tok.word_index[pair[0]]] = pair[1]

        # save the matrix to the dir
        np.save('./wt_'+dia+'/'+data_pair[0]+'/'+filep.split('.')[0]+'.npy', embedding_matrix.toarray())


def create_wt_my(datap):
    print('Working on: '+data_pair[1])
    print('Loading word vectors: '+data_pair[0])
    t = 'year'
    vecp = '/home/xiaolei/Documents/w2v/fasttext_cbow/'+datap[0]+'/'+t+'/'+datap[0]+'.vec'

    print('Loading tokenizer......')
    tok = pickle.load(open('./toks/'+data_pair[1]+'.pkl', 'rb'))

    opt_dir = './wt_my/'+data_pair[0]+'/'
    word_vectors = load_ft_vec(vecp)

    # loop through time domains    
    for dm in datap[2]:

        # create domain vocab
        vocab_dm = dict()
        for key in tok.word_index:
            vocab_dm[key+dm] = tok.word_index[key]

        
        print('Creating Embedding Matrix...............')
        # first create matrix
        vec_len = 300
        embedding_matrix = sparse.lil_matrix((len(vocab_dm) + 1, vec_len))

        for pair in word_vectors.items():
            # add the word if the word in the tokenizer
            if pair[0] in vocab_dm:
                embedding_matrix[vocab_dm[pair[0]]] = pair[1]

        # save the matrix to the dir
        np.save(opt_dir+dm+'.npy', embedding_matrix.toarray())

for data_pair in data_list:
    create_wt_my(data_pair)

for dia in dia_list:
    for data_pair in data_list:
        create_wt_ft(data_pair, dia)

