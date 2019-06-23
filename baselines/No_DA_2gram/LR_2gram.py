from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import pickle


def run_lr_3gram(data_name, train_path, test_path):
    """

    :param data_name:
    :type data_name: str
    :param train_path: training path
    :type train_path: str
    :param test_path: testing file path
    :type test_path: str
    :return:
    """
    print('Working on: '+data_name)
    # check if the vectorizer and exists
    # build the vectorizer
    if not (os.path.exists('./vects/' + data_name + '.pkl') and os.path.exists('./clfs/' + data_name + '.pkl')):

        print('Loading Training data........')
        # load the training data
        train_docs = []
        train_labels = []
        with open(train_path) as train_file:
            train_file.readline() # skip the 1st column names
            for line in train_file:
                if len(line.strip()) < 5:
                    continue
                
                infos = line.strip().split('\t')
                train_labels.append(int(infos[2]))
                train_docs.append(infos[0].strip())
        print(np.unique(train_labels))
        
        print('Fiting Vectorizer.......')
        vect = CountVectorizer(ngram_range=(1,2), max_features=15000, min_df=2)
        vect.fit(train_docs)
        pickle.dump(vect, open('./vects/'+data_name+'.pkl', 'wb')) # save the vectorizer

        print('Transforming Training data........')
        train_docs = vect.transform(train_docs)

        # encode the labels
#        le = LabelEncoder()
#        le.fit(train_labels)
#        pickle.dump(le, open('./vects/'+data_name+'.le', 'wb'))
#        le.transform(train_labels)

        # fit the model
        print('Building model............')
        if len(np.unique(train_labels)) > 2:
            clf = SGDClassifier(loss='log', class_weight='balanced') # , multi_class='ovr'
        else:
            clf = SGDClassifier(loss='log', class_weight='balanced')
        clf.fit(train_docs, train_labels)
        pickle.dump(clf, open('./clfs/' + data_name + '.pkl', 'wb'))  # save the classifier
    else:
        vect = pickle.load(open('./vects/'+data_name+'.pkl', 'rb'))
        clf = pickle.load(open('./clfs/'+data_name+'.pkl', 'rb'))

    # load the test data
    test_docs = []
    test_labels = []
    with open(test_path) as test_file:
        test_file.readline()
        for line in test_file:
            if len(line.strip()) < 5:
                continue
            infos = line.strip().split('\t')
            test_labels.append(int(infos[2]))
            test_docs.append(infos[0].strip())

    # transform the test data
    print('Testing.........')
    test_docs = vect.transform(test_docs)
    y_preds = clf.predict(test_docs)

    with open('results.txt', 'a') as writefile:
        writefile.write(data_name + '_________________\n')
        writefile.write(str(f1_score(y_pred=y_preds, y_true=test_labels, average='weighted'))+'\n')
        writefile.write(classification_report(y_pred=y_preds, y_true=test_labels, digits=3)+'\n')
        writefile.write('.........................\n')


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
        train_path = '../../split_data/'+data_name+'/train.tsv'
        test_path = '../../split_data/'+data_name+'/test.tsv'
        run_lr_3gram(data_name, train_path, test_path)
