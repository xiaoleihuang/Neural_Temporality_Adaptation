"""
This script is to create datasets for each baseline.
"""
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from multiprocessing import Pool
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np


baselines = [
    'DANN',
    'PBLM',
    'SCL_FLDA',
    'YeZhang',
]

data_list = [
#    'amazon_month',
    'amazon_year',
#    'dianping_month',
    'dianping_year',
#    'yelp_hotel_month',
    'economy_year',
    'parties_year',
#    'vaccine_month',
    'vaccine_year',
    'yelp_hotel_year',
#    'yelp_rest_month',
    'yelp_rest_year',
#    'economy_month',
]


def create_DANN_data_shallow(data_name):
    """
    This is to create data for DANN shallow network experiments
    :param data_name:
    :return:
    """
    print('Working on: ' + data_name)
    print('Build Toeknizer.......')
    # create a vectorizer for the data
    tokenizer = CountVectorizer(max_features=15000)
    raw_corpus = []
    # load the raw corpus
    with open('../raw_tsv_data/' + data_name + '.tsv') as raw_file:
        raw_file.readline()
        for line in raw_file:
            raw_corpus.append(line.split('\t')[0])
    tokenizer.fit(raw_corpus)

    # because the its svm format is based on indices
    data_dir = '../split_data/' + data_name + '/'
    # train is the source domain and the test is the target domain
    source_path = 'train.tsv'
    target_path = 'test.tsv'

    print('Creating the training dataset......')
    with open('./DANN/' + data_name + '_source.txt', 'w') as writefile:
        with open(data_dir + source_path) as source_file:
            source_file.readline()

            docs = []
            labels = []
            for line in source_file:
                infos = line.strip().split('\t')
                docs.append(infos[0])
                labels.append(infos[-1])

                if len(docs) >= 5000:
                    feas = tokenizer.transform(docs).toarray()
                    docs.clear()
                    for fea_idx in range(feas.shape[0]):
                        fea_pair = [
                            str(idx) + ':' + str(value)
                            for idx, value in enumerate(feas[fea_idx])
                            if value != 0
                        ]
                        writefile.write(
                            labels[fea_idx] + ' ' + ' '.join(fea_pair) + '\n'
                        )
                    labels.clear()

            if len(docs) > 0:
                feas = tokenizer.transform(docs).toarray()
                docs.clear()
                for fea_idx in range(feas.shape[0]):
                    fea_pair = [
                        str(idx) + ':' + str(value)
                        for idx, value in enumerate(feas[fea_idx])
                        if value != 0
                    ]
                    writefile.write(
                        labels[fea_idx] + ' ' + ' '.join(fea_pair) + '\n'
                    )
                labels.clear()

    print('Creating the testing dataset......')
    with open('./DANN/' + data_name + '_target.txt', 'w') as writefile:
        with open(data_dir + target_path) as target_file:
            target_file.readline()
            docs = []
            labels = []
            for line in target_file:
                infos = line.strip().split('\t')
                docs.append(infos[0])
                labels.append(infos[-1])

                if len(docs) >= 5000:
                    feas = tokenizer.transform(docs).toarray()
                    docs.clear()
                    for fea_idx in range(feas.shape[0]):
                        fea_pair = [
                            str(idx) + ':' + str(value)
                            for idx, value in enumerate(feas[fea_idx])
                            if value != 0
                        ]
                        writefile.write(
                            labels[fea_idx] + ' ' + ' '.join(fea_pair) + '\n'
                        )
                    labels.clear()

            if len(docs) > 0:
                feas = tokenizer.transform(docs).toarray()
                docs.clear()
                for fea_idx in range(feas.shape[0]):
                    fea_pair = [
                        str(idx) + ':' + str(value)
                        for idx, value in enumerate(feas[fea_idx])
                        if value != 0
                    ]
                    writefile.write(
                        labels[fea_idx] + ' ' + ' '.join(fea_pair) + '\n'
                    )
                labels.clear()


def create_DANN_data_deep(data_name):
    """
    This is to create data for deep CNN with flip layer
        1. Build tokenzier
        2. Split data into train and test data
    :return:
    """
    print('Working on: ' + data_name)
    print('Build Toeknizer.......')
    # create a vectorizer for the data
    tok = Tokenizer(num_words=15000)

    raw_corpus = []
    # load the raw corpus
    with open('../split_data/' + data_name + '/' + data_name + '.tsv') as raw_file:
        raw_file.readline()
        for line in raw_file:
            raw_corpus.append(line.split('\t')[0])
    tok.fit_on_texts(raw_corpus)

    # save the tokenizer to the file
    with open('./DANN/'+data_name+'.tok', 'wb') as tok_file:
        pickle.dump(tok, tok_file)

    # because the its svm format is based on indices
    data_dir = '../split_data/' + data_name + '/'
    # train is the source domain and the test is the target domain
    source_path = 'train.tsv'
    valid_path = 'valid.tsv'
    target_path = 'test.tsv'

    print('Creating the dataset......')
    path_pair = [
        (source_path, '_source.txt'),
        (valid_path, '_valid.txt'),
        (target_path, '_target.txt'),
    ]

    for paths in path_pair:
        with open('./DANN/' + data_name + paths[1], 'w') as writefile:
            with open(data_dir + paths[0]) as source_file:
                source_file.readline()

                docs = []
                labels = []
                domain_labels = []

                for line in source_file:
                    infos = line.strip().split('\t')
                    docs.append(infos[0])
                    labels.append(infos[-1])
                    domain_labels.append(infos[1])

                feas = tok.texts_to_sequences(docs)
                del docs

                # select the samples
                feas = pad_sequences(feas, 60)
                for pair in zip(labels, domain_labels, feas):
                    writefile.write(
                        str(pair[0]) + '\t' + str(pair[1]) + '\t' +
                        "\t".join([str(item) for item in pair[2]]) +'\n'
                    )

def DANN():
    # create the dir
    if not os.path.exists('./DANN'):
        os.mkdir('./DANN')

    # multiprocessing here
    with Pool(5) as p:
        p.map(create_DANN_data_deep, data_list)


def create_PBLM_data(data_name):
    print('Working on: ' + data_name)
    if not os.path.exists('./PBLM/' + data_name + '_source'):
        os.mkdir('./PBLM/' + data_name + '_source')
    if not os.path.exists('./PBLM/' + data_name + '_target'):
        os.mkdir('./PBLM/' + data_name + '_target')

    source_dir = './PBLM/' + data_name + '_source/'
    target_dir = './PBLM/' + data_name + '_target/'

    data_dir = '../split_data/' + data_name + '/'
    # train is the source domain and the test is the target domain
    source_path = 'train.tsv'
    target_path = 'test.tsv'

    source_un = open(source_dir + data_name + '_sourceUN.txt', 'w')
    source_neg_file = open(source_dir + 'negative.parsed', 'w')
    source_pos_file = open(source_dir + 'positive.parsed', 'w')

    target_un = open(target_dir + data_name + '_targetUN.txt', 'w')
    target_neg_file = open(target_dir + 'negative.parsed', 'w')
    target_pos_file = open(target_dir + 'positive.parsed', 'w')

    print('Creating source domain.....')
    with open(data_dir + source_path) as source_file:
        source_file.readline()  # skip the column name

        for line in source_file:
            infos = line.strip().split('\t')
            content = infos[0] + '\n'

            source_un.write(content)
            if int(infos[-1]) == 0:
                source_neg_file.write(content)
            else:
                source_pos_file.write(content)
    source_un.flush()
    source_un.close()
    source_neg_file.flush()
    source_neg_file.close()
    source_pos_file.flush()
    source_pos_file.close()

    print('Creating target domain....')
    with open(data_dir + target_path) as target_file:
        target_file.readline()  # skip the column name

        for line in target_file:
            infos = line.strip().split('\t')
            content = infos[0]+ '\n'
            target_un.write(content)
            if int(infos[-1]) == 0:
                target_neg_file.write(content)
            else:
                target_pos_file.write(content)
    target_un.flush()
    target_un.close()
    target_neg_file.flush()
    target_neg_file.close()
    target_pos_file.flush()
    target_pos_file.close()


def PBLM():
    if not os.path.exists('./PBLM'):
        os.mkdir('./PBLM')

    # multiprocessing here
    with Pool(3) as p:
        p.map(create_PBLM_data, data_list)


def create_SCL_FLDA_data(data_name):
    """

    :param data_name:
    :return:
    """
    print('Working on: ' + data_name)
    # the paper states they use the unigram and bigram features.
    print('Creating CountVectorizer.....')
    # create a vectorizer for the data
    tokenizer = CountVectorizer(max_features=15000, ngram_range=(1,3), min_df=2)
    raw_corpus = []
    # load the raw corpus
    with open('../split_data/' + data_name + '/' + data_name + '.tsv') as raw_file:
        raw_file.readline()
        for line in raw_file:
            raw_corpus.append(line.split('\t')[0])
    tokenizer.fit(raw_corpus)

    # because the its svm format is based on indices
    data_dir = '../split_data/' + data_name + '/'
    # train is the source domain and the test is the target domain
    source_path = 'train.tsv'
    target_path = 'test.tsv'

    print('Creating the training dataset......')
    with open('./SCL_FLDA/' + data_name + '_source.txt', 'w') as writefile:
        with open(data_dir + source_path) as source_file:
            source_file.readline()

            docs = []
            labels = []
            for line in source_file:
                infos = line.strip().split('\t')
                docs.append(infos[0])
                labels.append(infos[-1])

                if len(docs) >= 5000:
                    feas = tokenizer.transform(docs).toarray()
                    docs.clear()
                    for fea_idx in range(feas.shape[0]):
                        fea_pair = [
                            str(idx) + ':' + str(value)
                            for idx, value in enumerate(feas[fea_idx])
                            if value != 0
                        ]
                        writefile.write(
                            labels[fea_idx] + ' ' + ' '.join(fea_pair) + '\n'
                        )
                    labels.clear()

            if len(docs) > 0:
                feas = tokenizer.transform(docs).toarray()
                docs.clear()
                for fea_idx in range(feas.shape[0]):
                    fea_pair = [
                        str(idx) + ':' + str(value)
                        for idx, value in enumerate(feas[fea_idx])
                        if value != 0
                    ]
                    writefile.write(
                        labels[fea_idx] + ' ' + ' '.join(fea_pair) + '\n'
                    )
                labels.clear()

    print('Creating the testing dataset......')
    with open('./SCL_FLDA/' + data_name + '_' + 'target.txt', 'w') as writefile:
        with open(data_dir + target_path) as target_file:
            target_file.readline()
            docs = []
            labels = []
            for line in target_file:
                infos = line.strip().split('\t')
                docs.append(infos[0])
                labels.append(infos[-1])

                if len(docs) >= 5000:
                    feas = tokenizer.transform(docs).toarray()
                    docs.clear()
                    for fea_idx in range(feas.shape[0]):
                        fea_pair = [
                            str(idx) + ':' + str(value)
                            for idx, value in enumerate(feas[fea_idx])
                            if value != 0
                        ]
                        writefile.write(
                            labels[fea_idx] + ' ' + ' '.join(fea_pair) + '\n'
                        )
                    labels.clear()

            if len(docs) > 0:
                feas = tokenizer.transform(docs).toarray()
                docs.clear()
                for fea_idx in range(feas.shape[0]):
                    fea_pair = [
                        str(idx) + ':' + str(value)
                        for idx, value in enumerate(feas[fea_idx])
                        if value != 0
                    ]
                    writefile.write(
                        labels[fea_idx] + ' ' + ' '.join(fea_pair) + '\n'
                    )
                labels.clear()


def SCL_FLDA():
    if not os.path.exists('./SCL_FLDA'):
        os.mkdir('./SCL_FLDA')

    # multiprocessing here
    with Pool(1) as p:
        p.map(create_SCL_FLDA_data, data_list)


def create_YeZhang_data(data_name):
    print('Working on: ' + data_name)
    data_dir = '../split_data/' + data_name + '/'
    # train is the source domain and the test is the target domain
    source_path = 'train.tsv'
    target_path = 'test.tsv'

    source_neg_file = open('./YeZhang/' + data_name + '_source.neg', 'w')
    source_pos_file = open('./YeZhang/' + data_name + '_source.pos', 'w')

    target_neg_file = open('./YeZhang/' + data_name + '_target.neg', 'w')
    target_pos_file = open('./YeZhang/' + data_name + '_target.pos', 'w')

    print('Creating source domain.....')
    with open(data_dir + source_path) as source_file:
        source_file.readline()  # skip the column name

        for line in source_file:
            infos = line.strip().split('\t')
            if int(infos[-1]) == 0:
                source_neg_file.write(infos[0] + '\n')
            else:
                source_pos_file.write(infos[0] + '\n')

    source_neg_file.flush()
    source_neg_file.close()
    source_pos_file.flush()
    source_pos_file.close()

    print('Creating target domain....')
    with open(data_dir + target_path) as target_file:
        target_file.readline()  # skip the column name

        for line in target_file:
            infos = line.strip().split('\t')
            if int(infos[-1]) == 0:
                target_neg_file.write(infos[0] + '\n')
            else:
                target_pos_file.write(infos[0] + '\n')

    target_neg_file.flush()
    target_neg_file.close()
    target_pos_file.flush()
    target_pos_file.close()


def YeZhang():
    """
    Chinese corpus and non-sentiment corpus won't be used here, because we don't have the dictionary.
    :return:
    """
    if not os.path.exists('./YeZhang'):
        os.mkdir('./YeZhang')

    new_data_list = [
        item for item in data_list
        if 'dianping' in item and 'parties' not in item and 'economy' not in item
    ]

    # multiprocessing here
    with Pool(3) as p:
        p.map(create_YeZhang_data, new_data_list)


if __name__ == '__main__':
    DANN()
    #PBLM()
#    SCL_FLDA()
    # YeZhang()
    pass
