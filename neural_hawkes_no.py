'''
Test if diachronic word embedding works well or not
'''

import pickle
import os
import sys
import json

import gensim
from sklearn.metrics.classification import f1_score, classification_report
import numpy as np

import utils
import model_helper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # for Tensorflow cpu usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

"""
Define the data
"""
data_list = [
    # year
#    ('vaccine_year', './domain_weights/vaccine_year', './split_data_idx/vaccine_year/'),
#    ('economy_year', './domain_weights/economy_year', './split_data_idx/economy_year/'),
    ('yelp_rest_year', './domain_weights/yelp_rest_year', './split_data_idx/yelp_rest_year/'),
#    ('yelp_hotel_year', './domain_weights/yelp_hotel_year', './split_data_idx/yelp_hotel_year/'),
#    ('amazon_year', './domain_weights/amazon_year', './split_data_idx/amazon_year/'),
#    ('dianping_year', './domain_weights/dianping_year', './split_data_idx/dianping_year/'),
]

# load the configurations
config = utils.load_config('./config.ini')

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
            data['y'].append(int(line[1]))

    data['x'] = np.asarray(data['x'])
    data['y'] = np.asarray(data['y'])

    return data


def data_generator(data, dm, batch_size=64):
    '''
        structure of the data: dictionary of 'x'-list and 'y'-list
        
        dm: key for the domains
    '''
    # unique labels, 1 hot encoding
    uniq = len(np.unique(data['y']))
    
    steps = int(len(data['y']) / batch_size)
    if len(data['y']) % batch_size != 0:
        steps += 1

    for step in range(steps):
        batch_x = {}
        for key in dm:
            batch_x['input_'+key] = []
        batch_y = []
        
        for idx in range(step*batch_size, (step+1)*batch_size):
            if idx > len(data['y']) - 1:
                break

            for key in dm:
                batch_x['input_'+key].append(data['x'][idx])

            if uniq == 3:
                dlabel = [0]*3
                dlabel[data['y'][idx]] = 1
                batch_y.append(dlabel)
            else:
                batch_y.append(data['y'][idx])

        for key in dm:
            batch_x['input_'+key] = np.asarray(batch_x['input_'+key])
        batch_y = np.asarray(batch_y)

        yield batch_x, batch_y


def wt_generator(wt, dm):
    for idx, item in enumerate(dm):
        yield idx, item, wt


"""
Loop through each dataset
"""
for datap in data_list:
    print('Working on: -----------------'+datap[0])
    valid_result = open('./results/hawkes/results_noadapt.txt', 'a')

    ''' Check the number of time domain '''
    num_dm = set()
    with open('./split_data/'+datap[0]+'/'+datap[0]+'.tsv') as dfile:
        dfile.readline()
        for line in dfile:
            line = line.strip().split('\t')
            if len(line) != 3:
                continue
            
            num_dm.add(line[1])
    num_dm.add('general') # T + 1 general domain
    num_dm = list(sorted(num_dm))

    # load the matrix and extend to num_dm
    wt = np.load('./baselines/TEC/wt/'+datap[0]+'.npy')

    best_valid_f1 = 0
    wt_iter = wt_generator(wt, num_dm)

    # number of predicted labels
    if 'amazon' in datap[0] or 'yelp' in datap[0] or 'dianping' in datap[0]:
        config['rnn']['pred_num'] = 3
    else:
        config['rnn']['pred_num'] = 2
    hawkes_model = model_helper.create_hawkes(wt_iter, config['rnn'])
    valid_result.write(str(config['rnn']))

    # train, valid, test data
    dirp = './baselines/TEC/indices/'
    train_files = os.listdir(dirp)
    train_files = sorted([item for item in train_files if datap[0]+'.train' in item])
    train_data = {'x':[], 'y':[]}
    for filep in train_files:
        tmp_d = load_data(dirp+filep)
        train_data['x'].extend(tmp_d['x'])
        train_data['y'].extend(tmp_d['y'])

    valid_data = load_data(dirp+datap[0]+'.valid')
    test_data = load_data(dirp+datap[0]+'.test')

    for e in range(15):
        accuracy = 0.0
        loss = 0.0
        step = 1

        print('--------------Epoch: {}--------------'.format(e))

        # load data for each domain: data; time labels; labels
        train_iter = data_generator(train_data, num_dm, batch_size=64)

        # train on batches
        for x_train, y_train in train_iter:
            # skip only 1 class in the training data
            if len(np.unique(y_train)) == 1:
                continue

            # the data was padded in the iter
            tmp = hawkes_model.train_on_batch(
                x_train, {'senti':y_train},
                class_weight='auto',
            )

            # calculate loss and accuracy
            loss += tmp[0]
            loss_avg = loss / step
            accuracy += tmp[1]
            accuracy_avg = accuracy / step
            if step % 40 == 0:
                print('Step: {}'.format(step))
                print('\tLoss: {}. Accuracy: {}'.format(loss_avg, accuracy_avg))
                print('-------------------------------------------------')
            step += 1

        # valid on the validation set
        valid_iter = data_generator(valid_data, num_dm, batch_size=64)

        # evaluate by f1 score
        y_preds = []
        y_valids = []
        print('---------------------------Validation------------------------------')

        for x_valid, y_valid in valid_iter:
            tmp_preds = hawkes_model.predict(x_valid)
            
            for item_tmp in tmp_preds:
                y_preds.append(item_tmp)
            for item_tmp in y_valid:
                y_valids.append(item_tmp)

        y_preds = np.asarray(y_preds)
        y_valids = np.asarray(y_valids)
        if len(y_preds[0]) > 2:
            y_preds = np.argmax(y_preds, axis=1)
            y_valids = np.argmax(y_valids, axis=1)
        else:
            y_preds = [np.round(item[0]) for item in y_preds]

        valid_f1 = f1_score(y_true=y_valids, y_pred=y_preds, average='weighted')
        print('Validating f1-weighted score: ' + str(valid_f1))   
        
        if best_valid_f1 < valid_f1:
            best_valid_f1 = valid_f1
            best_model = hawkes_model
            test_iter = data_generator(test_data, num_dm, batch_size=64)
            y_preds_test = []
            y_tests = []
            print('------------------------------Test---------------------------------')
            for x_test, y_test in test_iter:
                tmp_preds = hawkes_model.predict(x_test)
                
                for item_tmp in tmp_preds:
                    y_preds_test.append(item_tmp)
                for item_tmp in y_test:
                    y_tests.append(item_tmp)

            y_preds_test = np.asarray(y_preds_test)
            y_tests = np.asarray(y_tests)
            if len(y_preds_test[0]) > 2:
                y_preds_test = np.argmax(y_preds_test, axis=1)
                y_tests = np.argmax(y_tests, axis=1)
            else:
                y_preds_test = [np.round(item[0]) for item in y_preds_test]

            valid_result.write(datap[0] + '\n')
            valid_result.write('Epoch ' + str(e) + '..................................................\n')
            valid_result.write('Valid f1: ' + str(valid_f1)+'\n')
            valid_result.write('\n')
            valid_result.write(
                'Test result: ' +
                str(f1_score(y_true=y_tests, y_pred=y_preds_test, average='weighted')) +
                '\n......................')
            valid_result.write(
                str(classification_report(y_true=y_tests, y_pred=y_preds_test, digits=3))+'\n')
            valid_result.write('\n')        
            valid_result.flush()
    valid_result.close()
