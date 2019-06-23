#coding:utf-8

from sklearn.feature_selection import mutual_info_classif
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def cal_mi(data_name, top_num=20):
    print('Working on: ' + data_name + '.........')
    data_path = '../raw_tsv_data/' + data_name + '.tsv'
    uniq_domains = set()
    domain_top_words = dict()
    top_words_trends = dict()
    
    universal_vect = TfidfVectorizer(min_df=2, ngram_range=(1,3), stop_words='english', )
    all_docs = []

    # get domain information
    with open(data_path) as data_file:
        # skip the 1st column line
        data_file.readline()
        for line in data_file:
            infos = line.strip().split('\t')
            if len(infos) != 3:
                continue
            uniq_domains.add(infos[1])
            all_docs.append(line.strip())

    # sample if the data is too large
    if len(all_docs) > 1000000:
        all_indices = list(range(len(all_docs)))
        np.random.seed(33)
        np.random.shuffle(all_indices)
        all_indices = all_indices[:1000000]
        all_docs = [all_docs[item] for item in all_indices]

    # 1st find out the top words in the final domain
    uniq_domains = sorted(uniq_domains)
    flag = True
    final_top_words = []
    
    print('Loop through each domain...............')
    # loop the documents again
    for dkey in uniq_domains:
        print('Working on domain: ' + str(dkey))
        domain_top_words[dkey] = dict()
        domain_docs = []
        domain_labels = []

        for line in all_docs:
            infos = line.strip().split('\t')
            if len(infos) != 3:
                continue
            if infos[1] != dkey:
                continue
            domain_docs.append(infos[0].strip())
            domain_labels.append(int(infos[2].strip()))
    
        print('Fitting Vectorizer....')
        domain_vect = TfidfVectorizer(min_df=2, ngram_range=(1,3), stop_words='english', max_features=15000)
        domain_docs = domain_vect.fit_transform(domain_docs)
        
        print('Calculate MI....')
        mi_score = mutual_info_classif(domain_docs, domain_labels)
        top_score_indices = list(reversed(mi_score.argsort()))[:top_num]
        
        # lookup dictionary of features and record the top mi for the domain
        domain_top_words[dkey] = []
        for word_key in domain_vect.vocabulary_:
            if domain_vect.vocabulary_[word_key] in top_score_indices:
                domain_top_words[dkey].append((word_key, mi_score[domain_vect.vocabulary_[word_key]]))
        
        top_words_trends[dkey] = dict()
        # find the trend of the top words
        if flag: # final domain
            top_words = [pair[0] for pair in domain_top_words[dkey]]
            flag = False

        top_words_trends[dkey] = dict()
        for tp_word in top_words:
            if tp_word in domain_vect.vocabulary_:
                top_words_trends[dkey][tp_word] = mi_score[domain_vect.vocabulary_[tp_word]]
            else:
                top_words_trends[dkey][tp_word] = 0.0
        
    print(domain_top_words)
    # save the results to the file
    with open('./mi/'+data_name+'_tops.pkl', 'wb') as write_file:
        pickle.dump(domain_top_words, write_file)
    with open('./mi/'+data_name+'_trends.pkl', 'wb') as write_file:
        pickle.dump(top_words_trends, write_file)


def viz_mi_trends(data_pair):
    data_name = data_pair[0]
    # presettings
    a4_dims = (15.7, 14.27)
    if 'dianping' in data_name:
        sns.set(font='Microsoft YaHei')
        import matplotlib as mpl
        font_name = 'Microsoft YaHei'
        mpl.rcParams['font.family']=font_name
    fig, ax = plt.subplots(figsize=a4_dims)
    

    # load the trends
    word_trend = pickle.load(open('./mi/'+data_name+'_trends.pkl', 'rb'))
    df = pd.DataFrame.from_dict(word_trend)
    pic = sns.heatmap(df, ax=ax, annot=True, cmap="YlGnBu", annot_kws={"size": 18}, cbar=False)
#    plt.tight_layout()
    plt.yticks(rotation = 15, fontsize=20)
    plt.title(data_pair[1], fontsize=20)
    plt.ylabel('Top words by MI', fontsize=20)
    plt.xticks([item + 0.45 for item in range(len(data_pair[2]))], data_pair[2], rotation=15, fontsize=20)
    
    plt.savefig('./mi_pic/'+data_name+'.pdf')
    
data_list = [
        ('vaccine_year', 'Twitter data - vaccine', ['2013', '2014', '2015', '2016']),
        ('amazon_month', 'Reviews data - music', ['Jan-Mar', 'Apr-Jun', 'Jul-Sep', 'Oct-Dec']),
        ('amazon_year', 'Reviews data - music', ['1997-99', '2000-02', '2003-05', '2006-08', '2009-11', '2012-14']),
        ('dianping_month', 'Reviews data - Chinese', ['Jan-Mar', 'Apr-Jun', 'Jul-Sep', 'Oct-Dec']),
        ('dianping_year', 'Reviews data - Chinese', ['2009', '2010', '2011', '2012']),
        ('economy_month', 'News data - economy', ['Jan-Mar', 'Apr-Jun', 'Jul-Sep', 'Oct-Dec']),
        ('economy_year', 'News data - economy', ['-1989', '1990-94', '1995-99', '2000-04', '2005-09', '2010-15']),
        ('parties_year', 'Politics data - platforms', ['1948-56', '1960-68', '1972-80', '1984-92', '1996-2004', '2008-16']),
        ('vaccine_month', 'Twitter data - vaccine', ['Jan-Mar', 'Apr-Jun', 'Jul-Sep', 'Oct-Dec']),
        ('yelp_hotel_month', 'Reviews data - hotel', ['Jan-Mar', 'Apr-Jun', 'Jul-Sep', 'Oct-Dec']),
        ('yelp_hotel_year', 'Reviews data - hotel', ['2005-08', '2009-11', '2012-14', '2015-17']),
        ('yelp_rest_month', 'Reviews data - restaurants', ['Jan-Mar', 'Apr-Jun', 'Jul-Sep', 'Oct-Dec']),
        ('yelp_rest_year', 'Reviews data - restaurants', ['2005-08', '2009-11', '2012-14', '2015-17']),
    ]

for pair in data_list:
#    cal_mi(pair[0])
    viz_mi_trends(pair)
