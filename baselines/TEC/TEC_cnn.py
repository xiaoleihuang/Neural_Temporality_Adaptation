'''
Recurrent CNN mode with contextual information
'''
import os
import sys

import gensim
from sklearn.metrics.classification import f1_score, classification_report
import numpy as np

from keras.layers import Input, Embedding, RepeatVector
from keras.layers import LSTM, Dense, Dropout, Lambda
from keras.layers import Conv1D, MaxPooling1D, Flatten
import keras
from keras.models import Model
import keras.backend as K
from keras import optimizers
import tensorflow as tf
import configparser

# for cpu use only
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    #for Tensorflow cpu usage
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

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


def load_data(filep):
    data = {'x': [], 'y':[]}
    with open(filep) as dfile:
        dfile.readline()

        for line in dfile:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            
            line[1] = int(line[1])
            data['x'].append(list(map(int, line[0].split())))
            data['y'].append(line[1])

    data['x'] = np.asarray(data['x'])
    data['y'] = np.asarray(data['y'])

    return data


def variable_with_weight_decay(name, initial_value, dtype=tf.float32, trainable=True, wd=None):
    """ Got from author open source codes
        Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        initial_value: initial value for Variable
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    var = tf.Variable(initial_value=initial_value, name=name, trainable=trainable, dtype=dtype)
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name=name + '_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def data_generator(data, batch_size=128):
    '''
        structure of the data: dictionary of 'x'-list and 'y'-list
        
        Because the RCNN requires left and right contexts of the words,
        therefore, this function will create two more contextual lists.
    '''
    # unique labels, 1 hot encoding
    uniq = len(np.unique(data['y']))
    
    steps = int(len(data['y']) / batch_size)
    if len(data['y']) % batch_size != 0:
        steps += 1

    for step in range(steps):
        batch_x = []
        left_x = []
        right_x = []
        batch_y = []
        
        for idx in range(step*batch_size, (step+1)*batch_size):
            if idx > len(data['y']) - 1:
                break

            batch_x.append(data['x'][idx])

            # generate contexts, obtain from author's source codes
            left_x.append([0] * len(data['x'][idx]))
            right_x.append([0] * len(data['x'][idx]))
            for offset, tid in enumerate(data['x'][idx]):
                if tid  == 0:
                    continue
                if offset == len(data['x'][idx])-1:
                    right_x[-1][offset-1] = tid
                    right_x[-1][offset] = tid
                else:
                    left_x[-1][offset+1] = tid
                    right_x[-1][offset-1] = tid

            if uniq == 3:
                dlabel = [0]*3
                dlabel[data['y'][idx]] = 1
                batch_y.append(dlabel)
            else:
                batch_y.append(data['y'][idx])
        
        batch_x = np.asarray(batch_x)
        batch_y = np.asarray(batch_y)
        left_x = np.asarray(left_x)
        right_x = np.asarray(right_x)
        right_x = np.flip(right_x, 1)

        yield batch_x, left_x, right_x, batch_y


def euclidean_distance_loss(params, params_prev):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param params: the current model parameters
    :param params_prev: previous model parameters
    :return: float
    """
    return K.sqrt(K.sum(K.square(params - params_prev), axis=-1))


def TEC_basic(config, f_prev=None, m_prev=None):
    '''
        config: parameter settings of the model
        f_prev: feature output from the model trained on the previous time domain
        m_prev: model params from the previous model
    '''
    wt_matrix = np.load('./wt/'+config['dname']+'.npy')

    # some model compile parameters
#    opt = keras.optimizers.SGD(.0001)
#    opt = keras.optimizers.RMSprop(.0001)
    opt = keras.optimizers.Adam(.0001)

    if config['pred_num'] == 3:
        pred_func = 'softmax'
        model_loss = {'pred':'categorical_crossentropy'}
    else:
        config['pred_num'] = 1
        pred_func = 'sigmoid'
        model_loss = {'pred':'binary_crossentropy'}
    
    # design inputs
    input_doc = Input(
        shape=(int(config['seq_max_len']),), 
        dtype='int32', name='input_doc',
    )
    input_left = Input(
        shape=(int(config['seq_max_len']),), 
        dtype='int32', name='input_left',
    )
    input_right = Input(
        shape=(int(config['seq_max_len']),), 
        dtype='int32', name='input_right',
    )
    # define inputs
    inputs = [input_doc, input_left, input_right]

    if f_prev:
        input_prev = Input(
            shape=(2*int(config['rnn_size']),), # output the same shape
            dtype='int32', name='input_prev'
        )

    # build embedding
    embed = Embedding(
        wt_matrix.shape[0],
        wt_matrix.shape[1],
        weights=[wt_matrix],
        input_length=int(config['seq_max_len']),
        trainable=False, # according to author open source codes
        name='embed'
    )
    
    embed_doc = embed(input_doc)
    embed_left = embed(input_left)
    embed_right = embed(input_right)

    # left and right are the contexts, connect with LSTM, reverse the right
    left_lstm = LSTM(wt_matrix.shape[1], name='left_lstm')(embed_left)
    left_lstm = RepeatVector(int(config['seq_max_len']))(left_lstm)
    right_lstm = LSTM(wt_matrix.shape[1], go_backwards=True, name='right_lstm')(embed_right)
    right_lstm = RepeatVector(int(config['seq_max_len']))(right_lstm)
    
    # concatenated
    concat = keras.layers.concatenate([left_lstm, embed_doc, right_lstm], axis=-1)
    
    # convolution
    conv = Conv1D(
        300, 3, strides=1, padding='valid', 
        activation='relu', use_bias=False, name='conv'
    )(concat)
    pool = MaxPooling1D(name='pool', strides=None, padding='valid')(conv)
    flatten = Flatten(name='flatten')(pool)

    # add f_prev if it is not None
    if f_prev:
        concat_f = keras.layers.concatenate([input_prev, flatten], axis=-1)
        
        # a dense layer with dropout
        concat_f = Dense(2*int(config['rnn_size']), activation='relu')(concat_f)
        concat_f = Dropout(0.5)(concat_f)

        # prediction
        pred = Dense(config['pred_num'], activation=pred_func, name='pred')(concat_f)
        
        # define inputs
        inputs.append(input_prev)

    else:
        # add a dropout
        f_dp = Dropout(0.5)(flatten)

        # prediction
        pred = Dense(config['pred_num'], activation=pred_func, name='pred')(f_dp)#'linear'

    # compile model
    my_model = Model(inputs=inputs, outputs=pred) 
    my_model.compile(
        loss=model_loss,
        optimizer=opt,
        metrics=['accuracy']
    )

    print(my_model.summary())
    return my_model


if __name__ == '__main__':
    # load the configurations
    config = load_config('../../config.ini')['rnn']
    config['epochs'] = 15

    data_list = [
        ('economy', 'economy_year'),
        ('vaccine', 'vaccine_year'),
        ('amazon', 'amazon_year'),
        ('dianping', 'dianping_year'),
        ('yelp_hotel', 'yelp_hotel_year'),
        ('yelp_rest', 'yelp_rest_year'),
    ]

    for datap in data_list:
        print('Working on: -----------------'+datap[0])
        valid_result = open('./results_cnn.txt', 'a')

        dirp = './indices/'
        files = os.listdir(dirp)
        config['dname'] = datap[1]

        '''train steps'''
        # load the train data, sorted to temporal order
        train_files = sorted([item for item in files if datap[1]+'.train' in item])
        valid_data = load_data(dirp+datap[1]+'.valid')

        # load and train data in an evolving way
        prev_model = None
        for order, filen in enumerate(train_files):
            tkey = filen.split('#')[1] # current time domain

            # load data
            train_data = load_data(dirp+filen)
            config['pred_num'] = len(np.unique(train_data['y']))

            # compile the model
            prev_model = None
            if os.path.exists('./models_cnn/'+datap[1]+'.h5'):
                prev_model = keras.models.load_model('./models_cnn/'+datap[1]+'.h5')
                # recompile the prev model to extrac intermediate outputs
                inter_model = Model(
                    inputs=[
                        prev_model.get_layer('input_doc').input, 
                        prev_model.get_layer('input_left').input,
                        prev_model.get_layer('input_right').input
                    ],
                    outputs=prev_model.get_layer('flatten').output
                )

                model = TEC_basic(config, f_prev='')
            else:
                model = TEC_basic(config, f_prev=None)

            best_model = None
            best_valid = 0.0

            # training steps
            print('-------------Training on Domain: ', datap[1], tkey)
            for e in range(20):
                accuracy = 0.0
                loss = 0.0
                step = 1

                train_iter = data_generator(train_data, batch_size=128)
                for x_train, x_left, x_right, y_train in train_iter:
                    # skip only 1 class in the training data
                    if len(np.unique(y_train)) == 1:
                        continue

                    if prev_model:
                        # obtain the intermediate output
                        inter_opt = inter_model.predict([x_train, x_left, x_right])
                        tmp = model.train_on_batch(
                            {
                                'input_doc':x_train, 'input_prev':inter_opt,
                                'input_left':x_left, 'input_right':x_right,
                            }, 
                            {'pred':y_train},
                            class_weight={'pred':'auto'},
                        )
                    else:
                        tmp = model.train_on_batch(
                            {
                                'input_doc':x_train, 'input_left':x_left, 
                                'input_right':x_right,
                            }, {'pred':y_train},
                            class_weight={'pred':'auto'},
                        )

                    # calculate loss and accuracy
                    loss += tmp[0]
                    loss_avg = loss / step
                    accuracy += tmp[1]
                    accuracy_avg = accuracy / step
                    if step % 10 == 0:
                        print('Step: {}'.format(step))
                        print('\tLoss: {}. Accuracy: {}'.format(loss_avg, accuracy_avg))
                        print('-------------------------------------------------')
                    step += 1

                '''validation and test step'''
                # evaluate by f1 score
                y_preds = []
                y_valids = []
                print('------------------------Validation--------------------------')

                valid_iter = data_generator(valid_data, batch_size=128)
                for x_valid, x_left, x_right, y_valid in valid_iter:
                    if prev_model:
                        # obtain the intermediate output
                        inter_opt = inter_model.predict([x_valid, x_left, x_right])
                        tmp_preds = model.predict(
                            {
                                'input_doc':x_valid, 'input_left':x_left, 
                                'input_right':x_right, 'input_prev':inter_opt
                            }
                        )
                    else:
                        tmp_preds = model.predict(
                            {'input_doc': x_valid, 'input_left':x_left, 
                            'input_right':x_right,}
                        )

                    for item_tmp in tmp_preds:
                        y_preds.append(item_tmp)
                    for item_tmp in y_valid:
                        y_valids.append(item_tmp)

                # convert to arrays
                y_preds = np.asarray(y_preds)
                y_valids = np.asarray(y_valids)

                if config['pred_num'] == 3:
                    y_preds = np.argmax(y_preds, axis=1)
                    y_valids = np.argmax(y_valids, axis=1)
                else:
                    y_preds = [np.round(item[0]) for item in y_preds]

                valid_f1 = f1_score(y_true=y_valids, y_pred=y_preds, average='weighted')
                print('Validating f1-weighted score: ' + str(valid_f1))   
                
                if best_valid < valid_f1:
                    best_valid = valid_f1
                    # save the current model
                    model.save('./models_cnn/'+datap[1]+'.h5')
                    best_model = model

            if order == len(train_files) - 1: # only use the last train domain
                '''test step'''
                print('---------------------test---------------------')
                # load test data
                test_data = load_data(dirp+datap[1]+'.test')
                test_data = data_generator(test_data, batch_size=128)

                y_preds = []
                y_tests = []

                for x_test, x_left, x_right, y_test in test_data:
                    if prev_model:
                        # obtain the intermediate output
                        inter_opt = inter_model.predict([x_test, x_left, x_right])
                        tmp_preds = best_model.predict(
                            {
                                'input_doc':x_test, 'input_left':x_left, 
                                'input_right':x_right, 'input_prev':inter_opt
                            }
                        )
                    else:
                        tmp_preds = best_model.predict(
                            {
                                'input_doc': x_test, 'input_left':x_left, 
                                'input_right':x_right,
                            }
                        )

                    for item_tmp in tmp_preds:
                        y_preds.append(item_tmp)
                    for item_tmp in y_test:
                        y_tests.append(item_tmp)

                if config['pred_num'] == 3:
                    y_preds = np.argmax(y_preds, axis=1)
                    y_tests = np.argmax(y_tests, axis=1)
                else:
                    y_preds = [np.round(item[0]) for item in y_preds]
                # write results
                valid_result.write(datap[1] + '\t' + tkey + '\n')
                valid_result.write(
                    'Test result: ' +
                    str(f1_score(y_true=y_tests, y_pred=y_preds, average='weighted')) +
                    '\n......................')
                valid_result.write(
                    str(classification_report(y_true=y_tests, y_pred=y_preds, digits=3))+'\n')
                valid_result.write('\n')        
                valid_result.flush()

