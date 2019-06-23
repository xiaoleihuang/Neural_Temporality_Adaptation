'''
Calculate the basic stats of the data:
    1. average number of tokens per document;
    2. total number of tokens;
    3. total number of word types (unique words, vocabulary);
'''
import json

if __name__ == '__main__':
    data_list = [
#        'vaccine',
#        'amazon',
        'economy',
#        'yelp_rest',
#        'yelp_hotel',
#        'dianping',
    ]
    stats = dict()
    mode = 'year'

    for dname in data_list:
        print('Working on: ', dname)
        stats[dname] = {
            'doc#tok': 0,
            'toks': 0,
            'vocab': set()
        }
        # load data
        with open('../raw_tsv_data/'+dname+'_'+mode+'.tsv') as dfile:
            dfile.readline() # skip the header
            for line in dfile:                
                line = line.strip()
                if len(line) < 2:
                    continue

                line = line.split('\t')
                if len(line) != 3:
                    continue

                line[1] = int(line[1])
                line[2] = int(line[2])

                words = [item for item in line[0].split() if len(item) > 0]
                stats[dname]['toks'] += len(words)
                stats[dname]['vocab'].update(words)
                stats[dname]['doc#tok'] += 1

        # process the stats
        stats[dname]['doc#tok'] = round(
            stats[dname]['toks']/stats[dname]['doc#tok'], 3
        )
        stats[dname]['vocab'] = len(stats[dname]['vocab'])

    print(stats)
#    with open('data_stats.json', 'w') as wfile:
#        wfile.write(json.dumps(stats, indent=4))
