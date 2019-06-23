'''
Convert split raw data into indices;

the data will be splitted into separated domain data
'''
import pickle
import os
from keras.preprocessing.sequence import pad_sequences


if __name__ == '__main__':
    data_list = [
        ('amazon', 'amazon_year'),
        ('dianping', 'dianping_year'),
        ('economy', 'economy_year'),
        ('vaccine', 'vaccine_year'),
        ('yelp_hotel', 'yelp_hotel_year'),
        ('yelp_rest', 'yelp_rest_year'),
    ]

    for datap in data_list:
        # load tokenizer
        tok = pickle.load(open('./toks/'+datap[1]+'.pkl', 'rb'))

        # load data        
        dirp = '../../split_data/'+datap[1]+'/'
        files = os.listdir(dirp)
        files = [item for item in files if item in [
            'train.tsv', 'valid.tsv', 'test.tsv']]

        # convert through files
        for filep in files:
            data = dict()
            with open(dirp+filep) as dfile:
                dfile.readline()
                for line in dfile:
                    line = line.strip().split('\t')
                    if len(line) != 3:
                        continue
                    if line[1] not in data:
                        data[line[1]] = {'x':[], 'y':[]}

                    data[line[1]]['x'].append(line[0])
                    data[line[1]]['y'].append(line[2])
            
            filen = filep.split('.')[0] # train, valid or test

            # loop through the key to create the indices
            for key in data:
                docs = pad_sequences(
                    tok.texts_to_sequences(data[key]['x']), 
                    maxlen=60, value=0
                )

                # define file name, then save the indices into file
                if filen == 'train':
                    wfn = './indices/'+datap[1]+'.'+filen+'#'+key
                else:
                    wfn = './indices/'+datap[1]+'.'+filen
                with open(
                    wfn, 'w'
                ) as wfile:
                    for idx, doc in enumerate(docs):
                        wfile.write(
                            ' '.join(map(str, doc))+'\t'+data[key]['y'][idx]+'\n')
                

