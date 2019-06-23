"""
This script undersamples data according to domains
"""
import pandas as pd
from collections import Counter

file_list = [
    # '../data/vaccine/vaccine_month.tsv',
    # '../data/vaccine/vaccine_year.tsv',
    '../data/amazon/amazon_review_month.tsv',
    '../data/amazon/amazon_review_year.tsv',
    '../data/yelp/yelp_Hotels_month.tsv',
    '../data/yelp/yelp_Hotels_year.tsv',
    '../data/yelp/yelp_Restaurants_month.tsv',
    '../data/yelp/yelp_Restaurants_year.tsv',
    # '../data/parties/parties_year.tsv',
    # '../data/aware/aware_month.tsv',
    # '../data/economy/economy_rel_month.tsv',
    # '../data/economy/economy_rel_year.tsv',
]

# load dataset
for file_path in file_list:
    print('Running File: ' + file_path)
    if file_path.endswith('tsv'):
        df = pd.read_csv(file_path, sep='\t')
    else:
        df = pd.read_csv(file_path)
    print(file_path)
    # find minimum length of domains
    counts = Counter(df.time)
    if 'month' in file_path and 'Hotels' not in file_path:
        min_val = int(min(counts.values())/10)
    else:
        min_val = int(min(counts.values()))

    new_path = file_path[:-4] + '_sample.tsv'
    # rewrite the data to the file
    with open(new_path, 'w') as writefile:
        writefile.write('content\ttime\tlabel\n')

    with open(new_path, 'a') as writefile:
        for domain in sorted(counts.keys()):
            tmp_df = df[df.time == domain]
            old_len = len(tmp_df)
            if len(tmp_df) > min_val:
                tmp_df = tmp_df.sample(min_val, random_state=33)
            new_len = len(tmp_df)

            print('Domain ' + str(domain) + " changes from old-->"+str(old_len)+" to new-->"+str(new_len))
            tmp_df.to_csv(writefile, header=False, sep='\t', index=False)
            tmp_df = None
    df = None
    print('--------------------------------------------')
