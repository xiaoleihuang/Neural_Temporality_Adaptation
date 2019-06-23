"""
This script is to create initial weights of embedding (lookup table)
The initial weights will be save as NPY format using numpy
"""
import utils
import pickle
import numpy as np
from multiprocessing import Pool


def create_domain_weigths(config_pair):
    """

    :param model_path:
    :param tkn_path:
    :param opt:
    :return:
    """
    vec_path, tkn_path, opt = config_pair
    print('Working on pre-trained embedding: '+str(vec_path))
    # load tokenizer
    with open(tkn_path, 'rb') as tkn_file:
        tkn = pickle.load(tkn_file)

    # loop through each domain
    for domain in tkn:
        print('Working on domain: '+str(domain))
        # get the vector generator
        vec_generator = utils.load_w2v(vec_path)
        tmp_w, tmp_v = next(vec_generator)
        print('Embedding size: ' + str(len(tmp_v)))
        
        embed_len = len(tkn[domain].word_index)
        if embed_len > tkn[domain].num_words:
            embed_len = tkn[domain].num_words

        embedding_matrix = np.zeros((embed_len + 1, len(tmp_v)))

        # add the word if the word in the tokenizer
#        if tmp_w in tkn[domain].word_index:
#            if tkn[domain].word_index[tmp_w] < tkn[domain].num_words:
#                embedding_matrix[tkn[domain].word_index[tmp_w]] = tmp_v
        # loop through each word vectors
        for word, vectors in vec_generator:
            if word in tkn[domain].word_index:
                if tkn[domain].word_index[word] < tkn[domain].num_words:
                    embedding_matrix[tkn[domain].word_index[word]] = vectors

        # save the matrix to the dir
        np.save(opt+'weights#'+str(domain)+'.npy', embedding_matrix)

if __name__ == '__main__':
    # a list of processing files,
    # format: vector model path, tokenizer path, output directory path
    filelist = [
        # amazon
        ('./fasttext/amazon/year/amazon.vec', './domain_tokenizer/amazon_year.tkn', './domain_weights/amazon_year/'),
        # yelp-hotel
        ('./fasttext/yelp/hotel/year/yelp_hotel.vec', './domain_tokenizer/yelp_hotel_year.tkn', './domain_weights/yelp_hotel_year/'),
        # yelp-rest
        ('./fasttext/yelp/rest/year/yelp_rest.vec', './domain_tokenizer/yelp_rest_year.tkn', './domain_weights/yelp_rest_year/'),
        # dianping
        ('./fasttext/dianping/year/dianping.vec', './domain_tokenizer/dianping_year.tkn', './domain_weights/dianping_year/'),
        # vaccine
        ('./fasttext/vaccine/year/vaccine.vec', './domain_tokenizer/vaccine_year.tkn', './domain_weights/vaccine_year/'),
        # economy
        ('./fasttext/economy/year/economy.vec', './domain_tokenizer/economy_year.tkn', './domain_weights/economy_year/'),
    ]
    
    p = Pool(5)
    p.map(create_domain_weigths, filelist)
