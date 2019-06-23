"""
This script is to create domain corpus by appending domain label as the suffix to each word
"""

filelist = [
    ('./split_data/amazon_year/amazon_year.tsv', 'amazon_year'),
    ('./split_data/yelp_rest_year/yelp_rest_year.tsv', 'yelp_rest_year'),
    ('./split_data/yelp_hotel_year/yelp_hotel_year.tsv', 'yelp_hotel_year'),
    ('./split_data/economy_year/economy_year.tsv', 'economy_year'),
    ('./split_data/vaccine_year/vaccine_year.tsv', 'vaccine_year'),
    ('./split_data/dianping_year/dianping_year.tsv', 'dianping_year'),
]

def domain_data(filepair):
    print(filepair)
    with open('./domain_tsv_data/' +filepair[1]+'.tsv', 'w') as writefile:
        with open(filepair[0]) as datafile:
            writefile.write(datafile.readline())

            for line in datafile:
                infos = line.split('\t')
                content = " ".join([word.strip()+str(infos[1]) for word in infos[0].split() if len(word.strip()) > 0])
                # for the specific domain
                writefile.write(content + '\t' + infos[1] + '\t' + infos[2]) # because infos[2] contains new line character,
                writefile.write(line) # for general domain


from multiprocessing import Pool
with Pool(3) as p:
    p.map(domain_data, filelist)
