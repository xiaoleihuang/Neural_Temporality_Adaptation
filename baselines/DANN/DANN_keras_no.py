"""No adversarial training
"""

#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.layers import Input, Conv1D, Embedding, Dropout
from keras.layers import MaxPool1D, Dense, Flatten
from keras.models import Model
from utils_dann import flipGradientTF
import numpy as np
from sklearn.metrics.classification import f1_score
from sklearn.metrics import classification_report
# original paper: https://arxiv.org/pdf/1505.07818.pdf
# model reference: https://cloud.githubusercontent.com/assets/7519133/19722698/9d1851fc-9bc3-11e6-96af-c2c845786f28.png
import sys


data_list = [
    ('vaccine', 'vaccine_year'),
#    ('amazon', 'amazon_month'),
    ('amazon', 'amazon_year'),
#    ('dianping', 'dianping_month'),
    ('dianping', 'dianping_year'),
#    ('google', 'economy_month'),
#    ('google', 'economy_year'),
#    ('google', 'parties_year'),
#    ('vaccine', 'vaccine_month'),
#    ('yelp_hotel', 'yelp_hotel_month'),
    ('yelp_hotel', 'yelp_hotel_year'),
#    ('yelp_rest', 'yelp_rest_month'),
    ('yelp_rest', 'yelp_rest_year'),
#    ('economy', 'economy_year'),
#    ('economy', 'economy_month'),
]

def data_loader(data_name):
    train_path = './data/'+ data_name + '_source.txt'
    valid_path = './data/' + data_name + '_valid.txt'
    test_path = './data/' + data_name + '_target.txt'
    
    domain_data = []
    train_data = []
    valid_data = []
    test_data = []

    label_encoder = set()
    domain_encoder = set()

    for dpath in [train_path, valid_path, test_path]:
        with open(dpath) as dfile:
            dfile.readline()
            for line in dfile:
                line = line.strip()
                if len(line.strip()) < 5:
                    continue #  filter out blank lines

                line = line.split('\t')
                dlabel = [int(line[1])]
                label = [int(line[0])]
                line = [int(item) for item in line[2:]]

                label_encoder.add(label[0])
                domain_encoder.add(dlabel[0])

                if dpath == train_path:
                    train_data.append(label+line)
                if dpath == test_path:
                    test_data.append(label+line)
                if dpath == valid_path:
                    valid_data.append(label+line)
                if dpath in [train_path, valid_path]:
                    domain_data.append(dlabel + line)

    return domain_data, train_data, valid_data, test_data, label_encoder, domain_encoder


def data_gen(docs, batch_size=64):
    """
        Batch generator
    """
    np.random.shuffle(docs) # random shuffle the training documents
    steps = int(len(docs) / batch_size)
    if len(docs) % batch_size != 0:
        steps += 1

    for step in range(steps):
        batch_docs = []
        batch_labels = []

        for idx in range(step*batch_size, (step+1)*batch_size):
            if idx > len(docs) -1:
                break
            batch_docs.append(np.asarray(docs[idx][1:]))
            batch_labels.append(int(docs[idx][0]))

        # convert to array
        batch_docs = np.asarray(batch_docs)
        batch_labels = np.asarray(batch_labels)

        yield batch_docs, batch_labels


def domain_data_gen(domain_docs, batch_size=64):
    """ Generate domain data
    """
    # load the data
    tmp_docs = np.random.choice(list(range(len(domain_docs))), size=batch_size, replace=False)
    tmp_docs = [domain_docs[idx] for idx in tmp_docs]

    batch_docs = {'domain_input': []}
    batch_labels = {'domain': []}

    for tmp_doc in tmp_docs:
        batch_docs['domain_input'].append(tmp_doc[1:])
        batch_labels['domain'].append(tmp_doc[0])
    
    return batch_docs, batch_labels


def run_dnn(data_pair):
    print('Working on: '+data_pair[1])
    wt_path = './weights/'+ data_pair[1] + '.npy'
    train_path = './data/'+ data_pair[1] + '_source.txt'
    valid_path = './data/' + data_pair[1] + '_valid.txt'
    test_path = './data/'+ data_pair[1] + '_target.txt'
    epoch_num = 15

    # parameters
    sent_len = 60 # the max length of sentence

    # load the data
    domain_data, train_data, valid_data, test_data, label_encoder, domain_encoder = data_loader(data_pair[1])

    label_encoder = list(sorted(label_encoder))
    domain_encoder = list(sorted(domain_encoder))

    """Preprocess"""
    # load weights
    weights = np.load(wt_path)

    # inputs
    text_input = Input(shape=(sent_len,), dtype='int32', name='text_input')
    domain_input = Input(shape=(sent_len,), dtype='int32', name='domain_input')

    # shared embedding
    embedding = Embedding(
        weights.shape[0], weights.shape[1], # size of data embedding
        weights=[weights], input_length=sent_len,
        trainable=True,
        name='embedding'
    )
    
    # shared CNN
    conv1 = Conv1D(
        filters=300,
        kernel_size=5,
        padding='valid',
        strides=1,
    )
    conv2 = Conv1D(
        filters=200,
        kernel_size=7,
        padding='valid',
        strides=1,
    )
    max_pool = MaxPool1D()
    flatten = Flatten()

    # start to share
    sent_embed = embedding(text_input)
    domain_embed = embedding(domain_input)

    sent_conv1 = conv1(sent_embed)
    domain_conv1 = conv1(domain_embed)

    sent_conv2 = conv2(sent_conv1)
    domain_conv2 = conv2(domain_conv1)

    sent_pool = max_pool(sent_conv2)
    domain_pool = max_pool(domain_conv2)
    
    sent_flat = flatten(sent_pool)
    domain_flat = flatten(domain_pool)

    # for sentiment clf
    dense_1 = Dense(128, activation='relu')(sent_flat)
    dense_dp = Dropout(0.2)(dense_1)

    # for domain prediction
    hp_lambda = 0.01

#    flip = flipGradientTF.GradientReversal(hp_lambda)(domain_flat)
    dense_da = Dense(128, activation='relu')(domain_flat)
    dense_da_dp = Dropout(0.2)(dense_da)
    da_preds = Dense(len(domain_encoder), activation='softmax', name='domain')(dense_da_dp) # multiple

    if 'dianping' in data_pair[1] or 'amazon' in data_pair[1] or 'yelp' in data_pair[1]:
        sentiment_preds = Dense(3, activation='softmax', name='senti')(dense_dp) # multilabels
        model_sent = Model(
            inputs=[text_input, domain_input], outputs=[sentiment_preds, da_preds],
        )
        model_sent.compile(
            loss={'senti': 'categorical_crossentropy', 'domain':'categorical_crossentropy'},
            loss_weights={'senti': 1, 'domain':0.005},
            optimizer='adam')
    else:
        sentiment_preds = Dense(1, activation='sigmoid', name='senti')(dense_dp) # binary
        model_sent = Model(
            inputs=[text_input, domain_input], outputs=[sentiment_preds, da_preds],
        )
        model_sent.compile(
            loss={'senti': 'binary_crossentropy', 'domain':'categorical_crossentropy'},
            loss_weights={'senti': 1, 'domain':0.005},
            optimizer='adam')

    print(model_sent.summary())
    best_valid_f1 = 0.0

    # fit the model
    for e in range(epoch_num):
        accuracy = 0.0
        loss = 0.0
        step = 1

        print('--------------Epoch: {}--------------'.format(e))

        train_iter = data_gen(train_data)
        # train sentiment
        # train on batches
        for x_train, y_train in train_iter:
            # skip only 1 class in the training data
            if len(np.unique(y_train)) == 1:
                continue

            batch_docs, batch_labels = domain_data_gen(domain_data, len(x_train))
            batch_docs['text_input'] = x_train

            # encoder the (domain) labels
            if len(label_encoder) > 2:
                y_train_tmp = []
                for idx in range(len(y_train)):
                    dlabel = [0]*len(label_encoder)
                    dlabel[label_encoder.index(y_train[idx])] = 1
                    y_train_tmp.append(dlabel)
                y_train = y_train_tmp

            dlabels = []
            for idx in range(len(batch_labels['domain'])):
                dlabel = [0]*len(domain_encoder)
                dlabel[domain_encoder.index(batch_labels['domain'][idx])] = 1
                dlabels.append(dlabel)

            batch_labels['domain'] = dlabels
            batch_labels['senti'] = y_train

            # convert to arrays
            for key in batch_docs:
                batch_docs[key] = np.asarray(batch_docs[key])
            for key in batch_labels:
                batch_labels[key] = np.asarray(batch_labels[key])

            # train sentiment model
            tmp_senti = model_sent.train_on_batch(
                batch_docs,
                batch_labels,
                class_weight={'senti:': 'auto', 'domain': 'auto'}
            )
            # calculate loss and accuracy
            loss += tmp_senti[0]
            loss_avg = loss / step
            if step % 40 == 0:
                print('Step: {}'.format(step))
                print('\tLoss: {}.'.format(loss_avg))
                print('-------------------------------------------------')
            step += 1

        # each epoch try the valid data, get the best valid-weighted-f1 score
        print('Validating....................................................')
        valid_iter = data_gen(valid_data)
        y_preds_valids = []
        y_valids = []
        for x_valid, y_valid in valid_iter:
            x_valid = np.asarray(x_valid)
            tmp_preds_valid = model_sent.predict([x_valid, x_valid])
            for item_tmp in tmp_preds_valid[0]:
                y_preds_valids.append(item_tmp)
            for item_tmp in y_valid:
                y_valids.append(int(item_tmp))

        if len(y_preds_valids[0]) > 2:
            y_preds_valids = np.argmax(y_preds_valids, axis=1)
        else:
            y_preds_valids = [np.round(item[0]) for item in y_preds_valids]

        f1_valid = f1_score(y_true=y_valids, y_pred=y_preds_valids, average='weighted')
        print('Validating f1-weighted score: ' + str(f1_valid))

        # if the validation f1 score is good, then test
        if f1_valid > best_valid_f1:
            best_valid_f1 = f1_valid
            test_iter = data_gen(test_data)
            y_preds = []
            y_tests = []
            for x_test, y_test in test_iter:
                x_test = np.asarray(x_test)
                tmp_preds = model_sent.predict([x_test, x_test])
                for item_tmp in tmp_preds[0]:
                    y_preds.append(item_tmp)
                for item_tmp in y_test:
                    y_tests.append(int(item_tmp))

            if len(y_preds[0]) > 2:
                y_preds = np.argmax(y_preds, axis=1)
            else:
                y_preds = [np.round(item[0]) for item in y_preds]

            test_result = open('./results_no.txt', 'a')
            test_result.write(data_pair[1] + '\n')
            test_result.write('Epoch ' + str(e) + '..................................................\n')
            test_result.write(str(f1_score(y_true=y_tests, y_pred=y_preds, average='weighted')) + '\n')
            test_result.write('#####\n\n')
            test_result.write(classification_report(y_true=y_tests, y_pred=y_preds, digits=3))
            test_result.write('...............................................................\n\n')


if __name__ == '__main__':
    for data_pair in data_list:
        run_dnn(data_pair)
