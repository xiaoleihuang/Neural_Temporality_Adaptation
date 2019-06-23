import pickle
import math
import os, sys, ast
import re
from . import model_helper

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from nltk.tokenize import TweetTokenizer
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

tokenizer = TweetTokenizer()
np.random.seed(33) # fix the random state for the sampling


def preprocess(tweet):
    """
    Preprocess a single tweet
    :param tweet:
    :return:
    """
    global tokenizer

    # lowercase
    tweet = tweet.lower()
    # replace url
    tweet = re.sub(r"https?:\S+", "URL", tweet)
    # replace user
    tweet = re.sub(r'@\w+', 'USER', tweet)
    # replace hashtag
    tweet = re.sub(r'#\S+', 'HASHTAG', tweet)
    # tokenize
    return " ".join([item.strip() for item in tokenizer.tokenize(tweet) if len(item.strip())>0])


def load_data(data_path, col_names=['content', 'time', 'label']):
    """
    Load data, because the loaded data are well-preprocessed, 
    thus, we stopped the preprocess step here.
    
    :param data_path: the path of csv or tsv file
    :param col_names:
    :return: [id, content, time, label]
    """
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path, sep=',')
    else:
        df = pd.read_csv(data_path, sep='\t')
    #df = df[col_names]
    #df['label'] = df['label'].fillna('no')
    #df['content'] = df['content'].apply(lambda x:preprocess(str(x))) # preprocessing
    return df.values.tolist()


def stratified_split(dataset, test=0.2, label_idx=-1, cv_num=1):
    """
    random stratified split by label, will update for year later
    :param dataset: list of dataset
    :param test:
    :param label_idx:
    :param cv_num: the number of cross_validation
    :return:
    """
    sss = StratifiedShuffleSplit(n_splits=cv_num, test_size=test, random_state=0)
    y = [item[label_idx] for item in dataset]

    train_idx, test_idx = [], []

    for train, test in sss.split(dataset, y):
        train_idx.append(train)
        test_idx.append(test)
    return train_idx, test_idx


def shuffle_split_data(X, TRAIN_SPLIT=0.8, VALIDATION_SPLIT=0.1, TEST_SPLIT=0.1):
    """
    Shuffle and then split the data into training, validation, testing dataset.
    :param X: original datasource
    :param TRAIN_SPLIT: proportion of training set
    :param VALIDATION_SPLIT: proportion of validation set
    :param TEST_SPLIT: proportion of testing set
    :return: the indices of training, validation, testing set
    """
    # shuffle
    length = len(X)
    shuffle_indices = np.arange(length)
    np.random.seed(33)
    np.random.shuffle(shuffle_indices)

    train_idx = shuffle_indices[:int(TRAIN_SPLIT * length)]
    valid_idx = shuffle_indices[int(TRAIN_SPLIT * length):int((1 - TEST_SPLIT) * length)]
    test_idx = shuffle_indices[-int(TEST_SPLIT * length):]

    return train_idx, valid_idx, test_idx


def mytokenizer(text):
    return text.split()


def train_fvs_da(dataset, balance=False, outputfile='features', fea_type = 'tfidf'):
    """
    Extract Feature vector of dataset
    :param dataset:
    :param balance:
    :return:
    """
    label_raw = [item[-1] for item in dataset]
    label_encoder = LabelEncoder()
    label_raw = label_encoder.fit_transform(label_raw)

    # baseline
    # baseline with domain adaptation
    # dataset = np.asarray(dataset)
    label_raw = np.asarray(label_raw)

    if balance:
        # over sampling to balance data
        random_sampler = RandomOverSampler(random_state=0)
        dataset, label_raw = random_sampler.fit_sample(dataset, label_raw)
        outputfile = outputfile + '_bal'

    if len(dataset) < 15469:  # this number is length of "./yelp/yelp_Hotels_year_sample.tsv"
        base_vect = TfidfVectorizer(min_df=2, ngram_range=(1, 3))
    else:
        base_vect = TfidfVectorizer(min_df=2, tokenizer=mytokenizer)
    if fea_type == 'tfidf':
        da_vect = model_helper.DomainVectorizer_tfidf()
    else:
        da_vect = model_helper.DomainVectorizer_binary()
    # feature transformation
    fvs_base = base_vect.fit_transform([item[-3] for item in dataset])
    fvs_da = da_vect.fit_transform(dataset)

    results = dict()
    results['label_raw'] = label_raw
    results['doc_raw'] = dataset
    results['label_encoder'] = label_encoder
    results['base_vect'] = base_vect
    results['da_vect'] = da_vect
    results['fvs_base'] = fvs_base
    results['fvs_da'] = fvs_da

    outputfile = outputfile+'_'+fea_type+'.pkl'
    pickle.dump(results, open(outputfile, 'wb'))
    return outputfile


def is_prime(num):
    """
    check whether a number is a prime
    :param num:
    :return:
    """
    if num % 2 == 0 and num > 2:
        return False
    for i in range(3, int(math.sqrt(num))+1, 2):
        if num % i == 0:
            return False
    return True


def add_suffix(tweet, domain):
    """
    Add suffix (time) to the end of word to explicitly show domain differences.
    :param tweet:
    :param domain:
    :return:
    """
    words = [tmp+'@'+str(domain) for tmp in tweet.split()]
    return tweet + " " + " ".join(words) # ??? create separate or not?


def append_domain_suffix(data_path, col_names=['content', 'time', 'score']):
    """
    This function add suffix (domain) to each word in the dataset, which is separated by '@'
    :param data_path: the orginal raw document
    :param col_names: the columns we want to extract
    :return:
    """
    # load the raw data
    df = pd.read_csv(data_path)
    df = df[col_names]
    df['intent'] = df['score'].fillna('no')
    df['content'] = df['content'].apply(lambda x: preprocess(str(x)))  # preprocessing

    df['content'] = df.apply(lambda row: add_suffix(row['content'], row['time']), axis=1)

    df.to_csv(os.path.basename(data_path)+'_suffix.csv')


def load_feature(filepath):
    if not os.path.isfile(filepath):
        print('File does not exist')
        sys.exit()

    feature_dicts = dict()
    with open(filepath) as datafile:
        uniq_domains = ast.literal_eval(datafile.readline())

        for line in datafile:
            infos = line.strip().split('\t')
            if str(infos[0]) not in feature_dicts:
                feature_dicts[str(infos[0])] = dict.fromkeys(uniq_domains, 0.0)
            feature_dicts[str(infos[0])][infos[1]] = float(infos[-1])
    return feature_dicts


def month2label(month):
    """
    Convert month to four season numbers
    :param month:
    :return:
    """
    if month in [1,2,3]:
        return 1
    elif month in [4,5,6]:
        return 2
    elif month in [7,8,9]:
        return 3
    else:
        return 4


def undersample(data_size, num):
    """
    This function return the sampled indices: shuffle the indices; take the top_num of the indices
    :param data_size:
    :param num:
    :return:
    """
    if data_size > num:
        indices = range(data_size)
        np.random.shuffle(indices)
        return indices[:num]
    else:
        return None


def undersample_file(file_path):
    """
    This script undersamples data according to domains
    :param file_path:
    :return:
    """
    if file_path.endswith('tsv'):
        df = pd.read_csv(file_path, sep='\t')
    else:
        df = pd.read_csv(file_path)

    from collections import Counter

    counts = Counter(df.time)
    min_val = min(counts.values())

    with open(file_path, 'w') as writefile:
        writefile.write('content\ttime\tlabel\n')

    with open(file_path, 'a') as writefile:
        for domain in counts.keys():
            tmp_df = df[df.time == domain]
            if len(tmp_df) > min_val:
                tmp_df = tmp_df.sample(min_val)
            tmp_df.to_csv(writefile, header=False)
