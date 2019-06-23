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

'''
Load weights
'''
def load_wt(dname, dia, mode='year'):
    klist = ['general']
    dirp = './dia_wt/'+dia+'/'+dname+'/'
    flist = os.listdir(dirp)
    for filep in flist:
        klist.append(filep.split('.')[0])

    for idx, key in enumerate(sorted(klist)):
        if key == 'general':
            wt_path = './domain_weights/'+dname+'_'+mode+'/weights#general.npy'
        else:
            wt_path = dirp+key+'.npy'

        yield idx, key, np.load(wt_path)


"""
Define the data
"""
data_list = [
    # year
    ('vaccine', './domain_weights/vaccine_year', './split_data_idx/vaccine_year/'),
    ('economy', './domain_weights/economy_year', './split_data_idx/economy_year/'),
    ('yelp_rest', './domain_weights/yelp_rest_year', './split_data_idx/yelp_rest_year/'),
    ('yelp_hotel', './domain_weights/yelp_hotel_year', './split_data_idx/yelp_hotel_year/'),
    ('amazon', './domain_weights/amazon_year', './split_data_idx/amazon_year/'),
    ('dianping', './domain_weights/dianping_year', './split_data_idx/dianping_year/'),
]

# load the configurations
config = utils.load_config('./config.ini')
dia_list = ['hamilton'] # 'kim', 'kulkarni', 'hamilton' 

"""
Loop through each dataset
"""
for dia in dia_list:
    for datap in data_list:
        print('Working on: -----------------'+datap[0])
        valid_result = open('./results/hawkes/results_rmsprop_dia.txt', 'a')

        best_valid_f1 = 0
        best_model = None
        wt_iter = load_wt(datap[0], dia)

        # number of predicted labels
        if 'amazon' in datap[0] or 'yelp' in datap[0] or 'dianping' in datap[0]:
            config['rnn']['pred_num'] = 3
        else:
            config['rnn']['pred_num'] = 2
        hawkes_model = model_helper.create_hawkes(wt_iter, config['rnn'])
        valid_result.write(str(config['rnn']))

        for e in range(15):
            accuracy = 0.0
            loss = 0.0
            step = 1

            print('--------------Epoch: {}--------------'.format(e))

            # load data for each domain: data; time labels; labels
            train_iter = utils.load_data(
                datap[2], #config['data']['data_dir']
                mode='train',
                batch_size=64,
                max_len=config['rnn']['seq_max_len'],
            )

            # train on batches
            for x_train, time_labels, y_train in train_iter:
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
            valid_iter = utils.load_data(
                datap[2],  # config['data']['data_dir']
                mode='valid',
                batch_size=config['rnn']['batch_size'],
                max_len=config['rnn']['seq_max_len'])

            # evaluate by f1 score
            y_preds = []
            y_valids = []
            print('---------------------------Validation------------------------------')

            for x_valid, y_valid in valid_iter:
                tmp_preds = hawkes_model.predict(x_valid)
                
                for item_tmp in tmp_preds:
                    y_preds.append(item_tmp)
                for item_tmp in y_valid:
                    y_valids.append(int(item_tmp))

            if len(y_preds[0]) > 2:
                y_preds = np.argmax(y_preds, axis=1)
            else:
                y_preds = [np.round(item[0]) for item in y_preds]

            valid_f1 = f1_score(y_true=y_valids, y_pred=y_preds, average='weighted')
            print('Validating f1-weighted score: ' + str(valid_f1))
            
            if best_valid_f1 < valid_f1:
                best_valid_f1 = valid_f1
                best_model = hawkes_model
                test_iter = utils.load_data(
                    datap[2],  # config['data']['data_dir']
                    mode='test',
                    batch_size=config['rnn']['batch_size'],
                    max_len=config['rnn']['seq_max_len'])
                y_preds_test = []
                y_tests = []

                print('------------------------------Test---------------------------------')
                for x_test, y_test in test_iter:
                    tmp_preds = hawkes_model.predict(x_test)
                    
                    for item_tmp in tmp_preds:
                        y_preds_test.append(item_tmp)
                    for item_tmp in y_test:
                        y_tests.append(int(item_tmp))

                if len(y_preds_test[0]) > 2:
                    y_preds_test = np.argmax(y_preds_test, axis=1)
                else:
                    y_preds_test = [np.round(item[0]) for item in y_preds_test]

                valid_result.write(datap[0] + '\t' + dia + '\n')
                valid_result.write('Epoch ' + str(e) + '..................................................\n')
                valid_result.write(
                    'Test result: ' +
                    str(f1_score(y_true=y_tests, y_pred=y_preds_test, average='weighted')) +
                    '\n......................')
                valid_result.write(
                    str(classification_report(y_true=y_tests, y_pred=y_preds_test, digits=3))+'\n')
                valid_result.write('\n')        
                valid_result.flush()
        valid_result.close()
