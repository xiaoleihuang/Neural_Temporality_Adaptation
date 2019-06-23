from utils import model_helper
import pickle
import os
import numpy as np
from scipy import sparse
from sklearn.metrics import f1_score, classification_report


data_dirs = [
    ('../../split_data/vaccine_year', 'vaccine_year'),
    #('../../split_data/amazon_month', 'amazon_month'),
    ('../../split_data/amazon_year', 'amazon_year'),
#    ('../../split_data/yelp_hotel_month', 'yelp_hotel_month'),
    ('../../split_data/yelp_hotel_year', 'yelp_hotel_year'),
    #('../../split_data/yelp_rest_month', 'yelp_rest_month'),
    ('../../split_data/yelp_rest_year', 'yelp_rest_year'),
    #('../../split_data/dianping_month', 'dianping_month'),
    ('../../split_data/dianping_year', 'dianping_year'),
#    ('../../split_data/economy_month', 'economy_month'),
    ('../../split_data/economy_year', 'economy_year'),
#    ('../../split_data/vaccine_month', 'vaccine_month'),
    ('../../split_data/parties_year', 'parties_year'),
]

params_list = {
        'amazon_month' : {'C': 3.0, 'l1_ratio': 0.1, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 30},
        'amazon_year' : {'C': 1.0, 'l1_ratio': 0.1, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 30},
        'economy_month' : {'C': 3, 'l1_ratio': 0, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 300},
        'economy_year' : {'C': 1, 'l1_ratio': 0.1, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 100},
        'parties_year' : {'C': 1, 'l1_ratio': 0.1, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 100},
        'vaccine_month' : {'C': 1, 'l1_ratio': 0, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 50},
        'vaccine_year' : {'C': 1, 'l1_ratio': 0.1, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 300},
        'yelp_hotel_month' : {'C': 1.0, 'l1_ratio': 0, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 30},
        'yelp_hotel_year' : {'C': 1, 'l1_ratio': 0.1, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 30},
        'yelp_rest_month' : {'C': 1.0, 'l1_ratio': 0, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 30},
        'yelp_rest_year' : {'C': 1.0, 'l1_ratio': 0.1, 'tol': 0.0001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 300},
        'dianping_year': {},
        'dianping_month' : {},
}

if not os.path.exists('./results.txt'):
    results = open('./results.txt', 'w')
else:
    results = open('./results.txt', 'a')

for data_dir in data_dirs:
    print('Working on: ' + data_dir[0])    
    
    if not os.path.exists('./clfs/'+data_dir[1]+'.model'):      
        print('Building DA vectorizer.................')
        if not os.path.exists('./vects/'+data_dir[1]+'.pkl'):
            print('Loading all dataset for building vectorizer.............')
            # load the data
            dataset = []
            for tmp_file in ['train.tsv', 'valid.tsv', 'test.tsv']:
                with open(data_dir[0] + '/'+ tmp_file) as data_file:
                    data_file.readline() # skip the 1st line
                    for line in data_file:
                        dataset.append(line.strip().split('\t')[:2])
                if tmp_file == 'train.tsv':
                    # training data size: 200000
                    if len(dataset) > 200000:
                        np.random.seed(33)
                        indices = list(range(len(dataset)))
                        np.random.shuffle(indices)
                        indices = indices[:200000]
                        dataset = [dataset[idx_tmp] for idx_tmp in indices]
            da_vect = model_helper.DomainVectorizer_tfidf(1)
            da_vect.fit(dataset)
            del dataset
            print('Save the vectorizer..............')
            pickle.dump(da_vect, open('./vects/'+data_dir[1]+'.pkl', 'wb'))
        else:
            print('Loading vectorizer..........')
            da_vect = pickle.load(open('./vects/'+data_dir[1]+'.pkl', 'rb'))

        print('Loading training data.............')
        train_data = []
        train_label = []
        with open(data_dir[0] + '/'+ 'train.tsv') as data_file:
            data_file.readline()
            for line in data_file:
                infos = line.strip().split('\t')
                train_data.append([infos[0], infos[1]])
                train_label.append(int(infos[2]))
        
        # training data size: 200000
        if len(train_data) > 200000:
            np.random.seed(33)
            indices = list(range(len(train_data)))
            np.random.shuffle(indices)
            indices = indices[:200000]
            train_data = [train_data[idx_tmp] for idx_tmp in indices]
            train_label = [train_label[idx_tmp] for idx_tmp in indices]
        
        print('Transforming training data...........')        
        train_data = da_vect.transform(train_data)
        
        print('Building classifier')
        clf = model_helper.build_lr_clf()
        clf.fit(train_data, train_label)
        del train_data
        del train_label
        pickle.dump(clf, open('./clfs/'+data_dir[1]+'.pkl', 'wb'))
    else:
        # load clf
        clf = pickle.load(open('./clfs/'+data_dir[1]+'.pkl', 'rb'))

    general_len = -1 * len(da_vect.tfidf_vec_da['general'].vocabulary_)
    best_lambda = 1
    best_valid = 0
    lambda_list = [0.3, 1, 10, 30, 100, 300]
    print('Loading Valid data')
    valid_data = []
    valid_label = []
    with open(data_dir[0] + '/valid.tsv') as valid_file:
        valid_file.readline()
        for line in valid_file:
            infos = line.strip().split('\t')
            valid_data.append(infos)
            valid_label.append(int(infos[-1]))
    print('Transforming valid data....................')
    valid_data = da_vect.transform_test(valid_data)
    # for using only general features
    
    valid_data = sparse.lil_matrix(valid_data)
    # because the general features were appended finally, previous features are all domain features.
    valid_data[:, :general_len] = 0

    for lambda_item in lambda_list:
        exp_data = valid_data * lambda_item
        pred_label = clf.predict(exp_data)
        report_da = f1_score(y_true=valid_label, y_pred=pred_label, average='weighted')
        if report_da > best_valid:
            best_valid = report_da
            best_lambda = lambda_item

    # release memory
    del exp_data
    del valid_data
    del valid_label

    # load test
    print('Loading Test data')
    test_data = []
    test_label = []
    with open(data_dir[0] + '/'+ 'test.tsv') as test_file:
        test_file.readline()
        for line in test_file:
            infos = line.strip().split('\t')
            test_data.append([infos[0], infos[1]])
            test_label.append(int(infos[2]))

    print('Transforming test data....................')
    test_data = da_vect.transform_test(test_data)
    # for using only general features
    general_len = -1 * len(da_vect.tfidf_vec_da['general'].vocabulary_)
    test_data = sparse.lil_matrix(test_data)
    test_data[:, :general_len] = 0
    if 'lambda' in params_list[data_dir[1]]:
        test_data = test_data * params_list[data_dir[1]]['lambda']
    else:
        test_data = test_data*30
    
    print('Testing.............................')
    pred_label = clf.predict(test_data)
    report_da = f1_score(y_true=test_label, y_pred=pred_label, average='weighted')

    results.write(data_dir[1] + ':' + str(report_da)+ '\n')
    results.write(classification_report(y_true=test_label, y_pred=pred_label, digits=3))
    results.write('...............................\n')
    results.flush()

results.close()
