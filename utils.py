import numpy as np
import warnings
warnings.simplefilter(action='ignore')
import gensim
from keras.preprocessing.sequence import pad_sequences
from imblearn.over_sampling import RandomOverSampler
import glob
import os, sys
import pickle
import configparser
import tensorflow as tf
sampler = RandomOverSampler(random_state=33)


def load_config(path):
    """
    Load init settings
    :param path:
    :return:
    """
    config = configparser.ConfigParser()
    config.read(path)

    config = dict(config._sections)
    for sec in config.keys():
        for key in config[sec]:
            if config[sec][key].isdigit():
                config[sec][key] = int(config[sec][key])
            else:
                try:
                    config[sec][key] = float(config[sec][key])
                except:
                    pass
    return config


def load_google(vec_path):
    """
    Load word2vec trained by Google Word2vec
    :param vec_path:
    :return:
    """
    if vec_path.endswith('bin'):
        model = gensim.models.KeyedVectors.load_word2vec_format(vec_path, binary=True)
    else:
        model = gensim.models.KeyedVectors.load(vec_path)
    for pair in zip(model.wv.index2word, model.wv.syn0):
        yield pair[0], pair[1]


def load_glove(vec_path):
    """
    Load vectors trained by GloVe
    :param vec_path:
    :return:
    """
    with open(vec_path) as vec_file:
        for line in vec_file:
            tmp = line.strip().split()
            word = tmp[0]
            vectors = np.asarray(tmp[1:], dtype='float32')
            yield word, vectors


def load_fast(vec_path):
    """
    Load vectors trained by Fasttext
    :param vec_path:
    :return: a generator
    """
    with open(vec_path) as vec_file:
        # skip the 1st meta information
        vec_file.readline()
        for line in vec_file:
            tmp = line.strip().split()
            word = tmp[0]
            vectors = np.asarray(tmp[1:], dtype='float32')
            yield word, vectors


def load_w2v(vec_path):
    if 'fasttext' in vec_path:
        return load_fast(vec_path)
    if 'glove' in vec_path:
        return load_glove(vec_path)
    if 'google' in vec_path:
        return load_google(vec_path)

    return None


def balance_oversampling(X, y):
    """
    Balance the dataset by oversampling
    :param X:
    :param y:
    :return:
    """
    global sampler
    return sampler.fit_sample(X, y)


def pad_ctt(X, max_len=60, value=0):
    """

    :param X:
    :param max_len:
    :return:
    """
    new_X = pad_sequences(X, maxlen=max_len, value=value)
    return new_X


def load_data(dir_name, mode='train', batch_size=128, max_len=60):
    """
    Load data from the splitted idx data dir.

    :param dir_name:
    :param mode:
    :param batch_size:
    :return: A generator of data, which contains data, time labels, labels.
    :type: generator
    """
    print('Load data from: '+dir_name)
    if not dir_name.endswith('/'):
        dir_name = dir_name + '/'
    file_list = sorted(glob.glob(dir_name + mode + '*.tsv'))

    count = 0 # count data length
    uniq_da = set()
    uniq_label = set()
    with open(file_list[0]) as tmp_f:
        tmp_f.readline()
        for line in tmp_f:
            infos = line.strip().split('\t')
            uniq_da.add(infos[1])
            count += 1
            uniq_label.add(int(infos[-1]))

    uniq_da = len(uniq_da)
    uniq_label = list(sorted(uniq_label))
    count -= 1
    print("We have :" + str(count)+" lines.")

    step = int(count / batch_size)
    if count % batch_size != 0:
        step += 1

    # open the all files
    handlers = [open(filep) for filep in file_list]
    # skip the 1st line of column names
    for handle in handlers:
        handle.readline()

    while step > 0:
        domain_data = {}
        time_labels = []
        labels = []
        label_flag = False # flag of if the labels have been recorded

        # loop through each file
        for handle in handlers:
            tmp_data = list()

            cur_key = os.path.basename(handle.name)
            cur_key = cur_key.split('#')[1].split('.')[0]

            for _ in range(batch_size): # create a batch-size dataset
                line = handle.readline()
                if len(line) == 0: # when finish reading all lines, stop reading the file
                    break

                infos = line.strip().split('\t')

                if not label_flag: # if the labels were recorded in the previous handler, they won't be recorded again.
                    labels.append(int(infos[-1]))
                    if mode == 'train':
                        # one hot encoder to encode time label
                        tmp_tl = [0] * uniq_da
                        tmp_tl[int(infos[-1])] = 1
                        time_labels.append(tmp_tl)

                tmp_data.append([int(idx) for idx in infos[0].split() if len(idx.strip()) > 0])

            if not label_flag:
                if mode == 'train' and len(uniq_label) > 2:
                    # encode the multiclasses to one hot encoding
                    for idx in range(len(labels)):
                        dlabel = [0] * len(uniq_label)
                        dlabel[uniq_label.index(labels[idx])] = 1
                        labels[idx] = dlabel
                labels = np.asarray(labels)
            label_flag = True

            # padd the data
            tmp_data = pad_ctt(tmp_data, max_len)

            # over sampling only for training dataset
#            if mode == 'train' and len(np.unique(labels)) > 1:
#                time_labels = np.asarray(time_labels)
#                time_labels = time_labels[indices]
#                tmp_data = tmp_data[indices]
#                labels = labels[indices]

            # padding the data
            domain_data['input_' + str(cur_key)] = np.asarray(tmp_data)
        step -= 1
        
        if mode == 'train':
            yield domain_data, time_labels, labels
        else:
            yield domain_data, labels


def load_data_sig(dir_name, mode='train', batch_size=128, max_len=60, seed=0):
    """
    Load data from the splitted idx data dir.

    :param dir_name:
    :param mode:
    :param batch_size:
    :return: A generator of data, which contains data, time labels, labels.
    :type: generator
    """
    print('Load data from: '+dir_name)
    if not dir_name.endswith('/'):
        dir_name = dir_name + '/'
    file_list = sorted(glob.glob(dir_name + mode + '*.tsv'))

    count = 0 # count data length
    uniq_da = set()
    uniq_label = set()
    with open(file_list[0]) as tmp_f:
        tmp_f.readline()
        for line in tmp_f:
            infos = line.strip().split('\t')
            uniq_da.add(infos[1])
            count += 1
            uniq_label.add(int(infos[-1]))

    uniq_da = len(uniq_da)
    uniq_label = list(sorted(uniq_label))
    count -= 1
    print("We have :" + str(count)+" lines.")

    step = int(count / batch_size)
    if count % batch_size != 0:
        step += 1

    # open the all files
    handlers = [open(filep) for filep in file_list]
    # skip the 1st line of column names
    for handle in handlers:
        handle.readline()

    while step > 0:
        domain_data = {}
        time_labels = []
        labels = []
        label_flag = False # flag of if the labels have been recorded

        # loop through each file
        for handle in handlers:
            tmp_data = list()

            cur_key = os.path.basename(handle.name)
            cur_key = cur_key.split('#')[1].split('.')[0]

            for _ in range(batch_size): # create a batch-size dataset
                line = handle.readline()
                if len(line) == 0: # when finish reading all lines, stop reading the file
                    break

                infos = line.strip().split('\t')

                if not label_flag: # if the labels were recorded in the previous handler, they won't be recorded again.
                    labels.append(int(infos[-1]))
                    if mode == 'train':
                        # one hot encoder to encode time label
                        tmp_tl = [0] * uniq_da
                        tmp_tl[int(infos[-1])] = 1
                        time_labels.append(tmp_tl)

                tmp_data.append([int(idx) for idx in infos[0].split() if len(idx.strip()) > 0])

            if not label_flag:
                if mode == 'train' and len(uniq_label) > 2:
                    # encode the multiclasses to one hot encoding
                    for idx in range(len(labels)):
                        dlabel = [0] * len(uniq_label)
                        dlabel[uniq_label.index(labels[idx])] = 1
                        labels[idx] = dlabel
                labels = np.asarray(labels)
            label_flag = True

            # padd the data
            tmp_data = pad_ctt(tmp_data, max_len)

            # padding the data
            domain_data['input_' + str(cur_key)] = np.asarray(tmp_data)
        step -= 1

        if mode == 'test':
            # sample for significance analysis
            indices = list(range(len(labels)))
            # fix the seed
            np.random.seed(seed)
            indices = np.random.choice(indices, size=len(indices))
            for key in domain_data:
                domain_data[key] = np.asarray([domain_data[key][item] for item in indices])
            domain_data[key] = np.asarray([domain_data[key][item] for item in indices])
            labels = np.asarray([labels[item] for item in indices])

        if mode == 'train':
            yield domain_data, time_labels, labels
        else:
            yield domain_data, labels


def doc2topic(docs, doc_tkn, topic_tkn):
    """Convert the indices of documents to the topic word indices
    """
    results = np.asarray([
        topic_tkn.doc2idx(doc.split(), unknown_word_index=0) for doc in doc_tkn.sequences_to_texts(docs)
    ])
    results = pad_ctt(results) # value=-1
    return results


def load_weights(wt_dir):
    """
    This script is to load weights of different domains
    :param wt_dir:
    :type wt_dir: str
    :return:
    """
    if not wt_dir.endswith('/'):
        wt_dir = wt_dir + '/'

    file_list = sorted(os.listdir(wt_dir))

    # loop through each file and return a generator: index, domain, weights
    for idx, file_name in enumerate(file_list):
        yield idx, file_name.split('#')[-1].split('.')[0], np.load(wt_dir+file_name)


def load_topic_weights(tpath):
    tmodel = gensim.models.wrappers.DtmModel.load(tpath)
    domains = tmodel.lambda_.shape[-1]
    twt = {}

    for idx in range(domains):
        if idx == domains-1:
            yield idx, 'last', tmodel.lambda_[:, :, idx].T
        else:
            yield idx, idx, tmodel.lambda_[:, :, idx].T


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
