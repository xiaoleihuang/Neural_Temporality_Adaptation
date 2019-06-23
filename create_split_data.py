"""
This script is to split the raw data into three parts:
    1. training
    2. valid
    3. test
The splitting strategy is to take the half of last domain's data as the testing; half of the last domain as the valid data;
other domains as the training data.
"""
import pandas as pd
import numpy as np
import os
import random
from multiprocessing import Pool


def split_data(filep):
    """

    :param tsv_file:
    :param out_dir:
    :return:
    """
    tsv_file = './raw_tsv_data/'+filep
    out_dir = './split_data/'+filep.split('.')[0]
    print(tsv_file)
    print(out_dir)

    if not out_dir.endswith('/'):
        out_dir = out_dir + '/'

    data_list = dict()

    with open(tsv_file) as datafile:
        col_names = datafile.readline()

        for line in datafile:
            if len(line.strip()) < 3:
                continue
            infos = line.split('\t')

            if int(infos[1]) not in data_list:
                data_list[int(infos[1])] = list()
            data_list[int(infos[1])].append(line)
    print('Loading Data Finished!')

    # check the domain
    max_domain = list(sorted(data_list.keys()))[-1]
    # get the least size of sample in the domain, for down sampling
    sample_size = min([len(data_list[item]) for item in data_list])

    print('Writing data to Valid & Test file')
    with open(out_dir + 'test.tsv', 'w') as testfile:
        with open(out_dir + 'valid.tsv', 'w') as validfile:
            validfile.write(col_names)
            testfile.write(col_names)

            sample_idx = list(range(len(data_list[max_domain])))
            np.random.shuffle(sample_idx)

            if len(data_list[max_domain]) > sample_size * 2:
                sample_idx = sample_idx[:sample_size*2]

            data_list[max_domain] = [line for idx, line in enumerate(data_list[max_domain]) if idx in sample_idx]
            for line in data_list[max_domain][:len(sample_idx)//2]:
                validfile.write(line)
            for line in data_list[max_domain][len(sample_idx)//2:]:
                testfile.write(line)

    # remove the max domain 
    del data_list[max_domain]

    print('Writing data to Train file, 1st round')
    train_list = []
    with open(out_dir + 'train.tsv', 'w') as trainfile:
        for key in data_list:
            if len(data_list[key]) > sample_size:
                sample_idx = list(range(len(data_list[key])))
                np.random.shuffle(sample_idx)
                sample_idx = sample_idx[:sample_size]
                data_list[key] = [line for idx, line in enumerate(data_list[key]) if idx in sample_idx]
            for line in data_list[key]:
                trainfile.write(line)
        del data_list

    with open(out_dir + 'train.tsv') as trainfile:
        trainfile.readline()
        train_list = trainfile.readlines()
    print('Shuffle Training data')
    train_indices = list(range(len(train_list)))
    np.random.shuffle(train_indices)

    print('Writing data to Train file, final round')
    with open(out_dir + 'train.tsv', 'w') as trainfile:
        trainfile.write(col_names)
        for idx in train_indices:
            trainfile.write(train_list[idx])


if __name__ == '__main__':
    # get the list of data from the raw_data folder
    listfiles = os.listdir('./raw_tsv_data')
    
    # currently only focus on the year data
    listfiles = [item for item in listfiles if 'month' in item]
    # split each data into the split_data folder
    with Pool(3) as p:
        p.map(split_data, listfiles)
