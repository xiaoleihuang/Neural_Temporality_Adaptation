from keras.preprocessing.text import Tokenizer
import pickle
import os


def create_tokenizer(domain_corpus, opt='./domain.tkn'):
    """
    Create tokenizers for each training domain
    :param domain_corpus: The path file of domain corpus
    :return:
    """
    print('Working on: '+domain_corpus)
    train_domains = ['general']
    count = 0 # odd is for general tokenizer, and even is for domain tokenizer
    corpora = dict.fromkeys(train_domains, list()) # save the loaded texts

    print('Start to read file')
    with open(domain_corpus) as readfile:
        readfile.readline() # skip the 1st line of column names
        for line in readfile:
            infos = line.rstrip().split('\t')
            if count % 2 == 0:
                dm_id = int(infos[1])
                if dm_id not in corpora:
                    corpora[dm_id] = list()
                corpora[dm_id].append(infos[0])#line.rstrip().split('\t')[0]
            else:
                corpora['general'].append(infos[0])

            count += 1
    # the one should be dropped because it was treated for test and valid dataset;
    # which should be blind to the classifier
    """Currently, we don't drop the valid domain, it will be used to calculate the different loss"""
    # drop_domain = max([domain for domain in corpora.keys() if domain != 'general'])
    # del corpora[drop_domain]  # remove the test and valid domain

    print('We have the training domains: '+str(corpora.keys()))
    # build tokenizer for each of the train_domains
    domain_tkn = dict()

    print('Fit the corpus to each tokenizer')
    for key in corpora:
        domain_tkn[key] = Tokenizer(num_words=15000, split=' ')
        domain_tkn[key].fit_on_texts(corpora[key])

    print('Save and finish')
    with open(opt, 'wb') as writefile:
        pickle.dump(domain_tkn, writefile)
    return domain_tkn

if __name__ == '__main__':
    # read list of domain tsv files
    domain_dir = './domain_tsv_data/'
    filelist = os.listdir(domain_dir)

    # loop through each corpus file
    for filep in filelist:
        if filep.endswith('.tsv'):
            if 'year' in filep:
                create_tokenizer(domain_dir+filep, './domain_tokenizer/'+filep.split('.')[0]+'.tkn')
