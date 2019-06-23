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



data_list = [
#    ('vaccine', 'vaccine_year'),
#    ('amazon', 'amazon_month'),
    ('google', 'economy_year'),
#    ('google', 'parties_year'),
#    ('amazon', 'amazon_year'),
#    ('dianping', 'dianping_month'),
#    ('dianping', 'dianping_year'),
#    ('google', 'economy_month'),

#    ('vaccine', 'vaccine_month'),
#    ('yelp_hotel', 'yelp_hotel_month'),
#    ('yelp_hotel', 'yelp_hotel_year'),
#    ('yelp_rest', 'yelp_rest_month'),
#    ('yelp_rest', 'yelp_rest_year'),
#    ('economy', 'economy_month'),
]

# load data
def load_data_iter(filename, batch_size=64, train=True):
    domain_labels = []  # for encoding domain labels
    labels = []
    docs = []
    label_indices = {}

    with open(filename) as data_file:
        for idx, line in enumerate(data_file):
            if len(line.strip()) < 5:
                continue #  filter out blank lines
            infos = line.strip().split('\t')
            labels.append(int(infos[0]))
            docs.append([int(item) for item in infos[2:]])

            if train:
                domain_labels.append(int(infos[1]))  # domain label position
                
                if labels[-1] not in label_indices:
                    label_indices[labels[-1]] = []
                else:
                    label_indices[labels[-1]].append(idx)
    """
    if train:
        # downsample to balance training data
        min_val = min([len(label_indices[key]) for key in label_indices])
        sampled_indices = []
        for key in label_indices:
            if len(label_indices[key]) > min_val:
                sampled_indices.extend(np.random.choice(label_indices[key], size=min_val, replace=False))
            else:
                sampled_indices.extend(label_indices[key])
        # resample
        domain_labels = [domain_labels[idx] for idx in sampled_indices]
        docs = [docs[idx] for idx in sampled_indices]
        labels = [labels[idx] for idx in sampled_indices]
    """

    uniqs = list(sorted(set(domain_labels)))  # for one-hot encoding
    steps = int(len(docs) / batch_size)
    if len(docs) % batch_size != 0:
        steps += 1

    for idx in range(steps):
        batch_data = np.asarray(docs[idx*batch_size: (idx+1)*batch_size])
        batch_label = np.asarray(labels[idx*batch_size: (idx+1)*batch_size])

        if train:
            # check if need one-hot encoding for the class prediction
            if 'dianping' in filename or 'yelp' in filename or 'amazon' in filename:
                tmp_labels = [[0]*3 for _ in range(len(batch_label))]
                for idx, item in enumerate(batch_label):
                    tmp_labels[idx][item] = 1

                batch_label = tmp_labels
            
            batch_domain_label = domain_labels[idx*batch_size: (idx+1)*batch_size]
            tmp_labels = [[0] * len(uniqs) for _ in range(len(batch_domain_label))]
            for idx, tmp in enumerate(batch_domain_label):
                tmp_labels[idx][uniqs.index(tmp)] = 1
            batch_domain_label = tmp_labels

            yield batch_data, np.asarray(batch_domain_label), np.asarray(batch_label)
        else:
            yield batch_data, batch_label


def run_dnn(data_pair):
    print('Working on: '+data_pair[1])
    wt_path = './weights/'+ data_pair[1] + '.npy'
    train_path = './data/'+ data_pair[1] + '_source.txt'
    valid_path = './data/' + data_pair[1] + '_valid.txt'
    test_path = './data/'+ data_pair[1] + '_target.txt'
    epoch_num = 15

    # parameters
    sent_len = 60 # the max length of sentence

    """Preprocess"""
    # load weights
    weights = np.load(wt_path)

    # input
    text_input = Input(shape=(sent_len,), dtype='int32', name='text_input')

    # embedding
    embedding = Embedding(
        weights.shape[0], weights.shape[1], # size of data embedding
        weights=[weights], input_length=sent_len,
        trainable=False,
        name='embedding'
    )(text_input)

    # CNN
    conv1 = Conv1D(
        filters=300,
        kernel_size=3,
        padding='valid',
        strides=1,
    )(embedding)
    conv2 = Conv1D(
        filters=200,
        kernel_size=5,
        padding='valid',
        strides=1,
    )(conv1)
    max_pool = MaxPool1D()(conv2)

    flatten = Flatten()(max_pool)

    # for sentiment clf
    dense_1 = Dense(128, activation='relu')(flatten)
    dense_dp = Dropout(0.2)(dense_1)

    # for domain prediction
    hp_lambda = 0.01

    """Obtain the number of domain label"""
    da_num = set()
    with open(train_path) as data_file:
        for line in data_file:
            da_num.add(line.strip().split('\t')[1]) # domain label position

    flip = flipGradientTF.GradientReversal(hp_lambda)(flatten)
    dense_da = Dense(128, activation='relu')(flip)
    dense_da_dp = Dropout(0.2)(dense_da)
    da_preds = Dense(len(da_num), activation='softmax', name='domain')(dense_da_dp) # multiple

    if 'dianping' in data_pair[1] or 'amazon' in data_pair[1] or 'yelp' in data_pair[1]:
        sentiment_preds = Dense(3, activation='softmax', name='senti')(dense_dp) # multilabels
        model_sentiment = Model(
            inputs=[text_input], outputs=[sentiment_preds, da_preds],
        )
        model_sentiment.compile(
            loss={'senti': 'categorical_crossentropy', 'domain':'categorical_crossentropy'},
            loss_weights={'senti': 1, 'domain':0.01},
            optimizer='adam')
    else:
        sentiment_preds = Dense(1, activation='sigmoid', name='senti')(dense_dp) # binary
        model_sentiment = Model(
            inputs=[text_input], outputs=[sentiment_preds, da_preds],
        )
        model_sentiment.compile(
            loss={'senti': 'binary_crossentropy', 'domain':'categorical_crossentropy'},
            loss_weights={'senti': 1, 'domain':0.01},
            optimizer='adam')

    print(model_sentiment.summary())
    best_valid_f1 = 0.0

    # fit the model
    for e in range(epoch_num):
        accuracy = 0.0
        loss = 0.0
        step = 1

        print('--------------Epoch: {}--------------'.format(e))

        train_iter = load_data_iter(train_path)
        # train sentiment
        # train on batches
        for x_train, time_labels, y_train in train_iter:
            # skip only 1 class in the training data
            if len(np.unique(y_train)) == 1:
                continue

            if time_labels.shape[0] != y_train.shape[0]:
                continue

            # train sentiment model
            tmp_senti = model_sentiment.train_on_batch(
                x_train,
                {'senti': y_train, 'domain': time_labels},
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
        valid_iter = load_data_iter(valid_path, train=False)
        y_preds_valids = []
        y_valids = []
        for x_valid, y_valid in valid_iter:
            x_valid = np.asarray(x_valid)
            tmp_preds_valid = model_sentiment.predict(x_valid)
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
            test_iter = load_data_iter(test_path, train=False)
            y_preds = []
            y_tests = []
            for x_test, y_test in test_iter:
                x_test = np.asarray(x_test)
                tmp_preds = model_sentiment.predict(x_test)
                for item_tmp in tmp_preds[0]:
                    y_preds.append(item_tmp)
                for item_tmp in y_test:
                    y_tests.append(int(item_tmp))

            if len(y_preds[0]) > 2:
                y_preds = np.argmax(y_preds, axis=1)
            else:
                y_preds = [np.round(item[0]) for item in y_preds]

            test_result = open('./results.txt', 'a')
            test_result.write(data_pair[1] + '\n')
            test_result.write('Epoch ' + str(e) + '..................................................\n')
            test_result.write(str(f1_score(y_true=y_tests, y_pred=y_preds, average='weighted')) + '\n')
            test_result.write('#####\n\n')
            test_result.write(classification_report(y_true=y_tests, y_pred=y_preds, digits=3))
            test_result.write('...............................................................\n\n')
            test_result.flush()


if __name__ == '__main__':
    for data_pair in data_list:
        run_dnn(data_pair)
