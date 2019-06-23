import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.layers import LSTM, Bidirectional
from keras.layers import Input, Embedding, Dense
from keras.layers import Dropout
from keras.models import Model
from sklearn.metrics import f1_score, classification_report
import numpy as np
import keras
from imblearn.over_sampling import RandomOverSampler
# input > embedding > Bi-LSTM > dense > dropout > sigmoid


# load data
def load_data_iter(filename, batch_size=50, train=True):
    labels = []
    docs = []

    with open(filename) as data_file:
        for line in data_file:
            infos = line.strip().split('\t')
            labels.append(int(infos[0]))
            docs.append([int(item) for item in infos[2:]])

    if train:
        # convert label to one hot labels
        if 'dianping' in filename or 'yelp' in filename or 'amazon' in filename:
            tmp_labels = [[0]*3 for _ in range(len(labels))]
            for idx, item in enumerate(labels):
                tmp_labels[idx][item] = 1

            labels = tmp_labels

    steps = int(len(docs) / batch_size)
    if len(docs) % batch_size != 0:
        steps += 1    

    for idx in range(steps):
        batch_data = np.asarray(docs[idx*batch_size: (idx+1)*batch_size])
        batch_label = np.asarray(labels[idx*batch_size: (idx+1)*batch_size])

        yield batch_data, batch_label


def run_bilstm(data_name):
    """
    This file will use DANN's existing data

    """
    print('Working on: '+data_name)
    # load w2v weights for the Embedding
    weights = np.load(open('../DANN/weights/'+data_name+'.npy', 'rb'))

    text_input = Input(shape=(60, ), dtype='int32')
    embed = Embedding(
        weights.shape[0], weights.shape[1], # size of data embedding
        weights=[weights], input_length=60,
        trainable=True,
        name='embedding')(text_input)

    bilstm = Bidirectional(
                LSTM(100, 
                    kernel_initializer="glorot_uniform",
                    kernel_regularizer=keras.regularizers.l1_l2(
                        0, 
                        0.0001,
                    ),
                    dropout=0.2,
                    recurrent_activation='tanh',
                )
            )(embed)

    # dense
    dense_l = Dense(100, activation='relu')(bilstm)
    dp_l = Dropout(0.2)(dense_l)

    # output
    if 'yelp' in data_name or 'amazon' in data_name or 'dianping' in data_name:
        pred_l = Dense(3, activation='softmax')(dp_l)
        model = Model(inputs=text_input, outputs=pred_l)
        model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        pred_l = Dense(1, activation='sigmoid')(dp_l)
        model = Model(inputs=text_input, outputs=pred_l)
        model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())
    best_valid_f1 = 0.0

    # fit the model
    epoch_num = 15
    train_path = '../DANN/data/' + data_name + '_source.txt'
    valid_path = '../DANN/data/' + data_name + '_valid.txt'
    test_path = '../DANN/data/' + data_name + '_target.txt'

    for e in range(epoch_num):
        accuracy = 0.0
        loss = 0.0
        step = 1

        print('--------------Epoch: {}--------------'.format(e))

        train_iter = load_data_iter(train_path)
        # train sentiment
        # train on batches
        for x_train, y_train in train_iter:
            # skip only 1 class in the training data
            if len(np.unique(y_train)) == 1:
                continue
            
            # train sentiment model
            tmp_senti = model.train_on_batch(
                x_train, y_train,
                class_weight= 'auto'
            )
            # calculate loss and accuracy
            loss += tmp_senti[0]
            loss_avg = loss / step
            accuracy += tmp_senti[1]
            accuracy_avg = accuracy / step

            if step % 40 == 0:
                print('Step: {}'.format(step))
                print('\tLoss: {}.'.format(loss_avg))
                print('\tAccuracy: {}.'.format(accuracy_avg))
                print('-------------------------------------------------')
            step += 1

        # each epoch try the valid data, get the best valid-weighted-f1 score
        print('Validating....................................................')
        valid_iter = load_data_iter(valid_path, train=False)
        y_preds_valids = []
        y_valids = []
        for x_valid, y_valid in valid_iter:
            x_valid = np.asarray(x_valid)
            tmp_preds_valid = model.predict(x_valid)
            for item_tmp in tmp_preds_valid:
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
                tmp_preds = model.predict(x_test)
                for item_tmp in tmp_preds:
                    y_preds.append(item_tmp)
                for item_tmp in y_test:
                    y_tests.append(int(item_tmp))

            if len(y_preds[0]) > 2:
                y_preds = np.argmax(y_preds, axis=1)
            else:
                y_preds = [np.round(item[0]) for item in y_preds]

            test_result = open('./results.txt', 'a')
            test_result.write(data_name + '\n')
            test_result.write('Epoch ' + str(e) + '..................................................\n')
            test_result.write(str(f1_score(y_true=y_tests, y_pred=y_preds, average='weighted')) + '\n')
            test_result.write('#####\n\n')
            test_result.write(classification_report(y_true=y_tests, y_pred=y_preds, digits=3))
            test_result.write('...............................................................\n\n')
            test_result.flush()


if __name__ == '__main__':
    data_list = [
        'vaccine_year',
        'economy_year',
        'yelp_rest_year',
        'yelp_hotel_year',
        'amazon_year',
        'dianping_year',
    ]
    for data_name in data_list:
        run_bilstm(data_name)
