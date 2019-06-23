"""
Convert train.tsv; valid.tsv; test.tsv to indices by their corresponding tokenizers

This is the the prior step for embedding layer
"""
import pickle
import os


def self_func(tmp_list, writefile, tokenizer, domain):
    doc_idx = tokenizer.texts_to_sequences(
        [' '.join([word+str(domain) if domain != 'general' else word for word in doc.split('\t')[0].split()]) for doc in tmp_list])

    for i in range(len(doc_idx)):
        infos = tmp_list[i].split('\t')
        if len(infos) != 3:
            continue
        # write idx, time, label to the file
        writefile.write(' '.join(map(str, doc_idx[i])) + '\t' + infos[1] + '\t' + infos[2] + '\n')
    tmp_list.clear()


def word2idx(tsv_dir, tkn_path, opt='./split_data_idx/'):
    # create a sub-folder under the opt dir
    dir_name = [tmp_name for tmp_name in tsv_dir.split('/')
                if len(tmp_name) > 1]
    dir_name = dir_name[-1]
    print('Working on ' + dir_name + '---------------------------')

    try:
        os.mkdir(opt + dir_name)
    except: # if exist, it won't create the dir
        pass

    # in case that the dir does not end with /
    if not tsv_dir.endswith('/'):
        tsv_dir += '/'
    # load tokenizer
    with open(tkn_path, 'rb') as tkn_file:
        tkn = pickle.load(tkn_file)

    # loop through each domain
    for domain in tkn:
        print('Working on domain: ' + str(domain))
        # create domain specific indices files
        train_domain = open(opt+'/'+dir_name+'/train#'+str(domain)+'.tsv', 'w')
        valid_domain = open(opt + '/' + dir_name + '/valid#' + str(domain) + '.tsv', 'w')
        test_domain = open(opt + '/' + dir_name + '/test#' + str(domain) + '.tsv', 'w')

        for pair in zip(['train.tsv', 'valid.tsv', 'test.tsv'], [train_domain, valid_domain, test_domain]):
            with open(tsv_dir+pair[0]) as tmp_file:
                pair[1].write(tmp_file.readline())
                count = 1
                tmp_list = []
                for line in tmp_file:
                    tmp_list.append(line.strip())
                    if count % 2000 == 0:
                        self_func(tmp_list, pair[1], tkn[domain], domain)
                    count += 1
                # in case any lefts were not processed
                if len(tmp_list) > 0:
                    self_func(tmp_list, pair[1], tkn[domain], domain)
            pair[1].flush()
            pair[1].close()

    print('Finished')

if __name__ == '__main__':
    data_dir = './split_data/'
    tkn_dir = './domain_tokenizer/'
    dir_list = os.listdir(data_dir)
    for data_name in dir_list:
        if 'year' not in data_name:
            continue
        print(data_name)
        word2idx(
            data_dir + data_name,
            tkn_dir + data_name+'.tkn')
