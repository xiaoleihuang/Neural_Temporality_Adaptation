'''
Build tokenizer for the baseline model
'''
from keras.preprocessing.text import Tokenizer
import pickle
import os


if __name__ == '__main__':
    data_list = [
        # year
        ('vaccine_year', '../../split_data/vaccine_year/'),
        ('economy_year', '../../split_data/economy_year/'),
        ('yelp_rest_year', '../../split_data/yelp_rest_year/'),
        ('yelp_hotel_year', '../../split_data/yelp_hotel_year/'),
        ('amazon_year', '../../split_data/amazon_year/'),
        ('dianping_year', '../../split_data/dianping_year/'),
    ]

    for datap in data_list:
        print('Working on: ', datap[0])

        data = []
        # load the data from the whole dataset
        with open(datap[1] + datap[0] + '.tsv') as dfile:
            dfile.readline()
            for line in dfile:
                line = line.strip().split('\t')
                if len(line) != 3:
                    continue
                data.append(line[0])

        tok = Tokenizer(num_words=15000, split=' ')
        tok.fit_on_texts(data)

        # save
        pickle.dump(tok, open('./toks/'+datap[0]+'.pkl', 'wb'))
