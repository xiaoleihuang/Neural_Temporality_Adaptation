if [ ! -d "fasttext" ]; then
  mkdir fasttext
fi

# amazon
fasttext cbow -input raw_data/amazon.txt -output ./fasttext_cbow/amazon/amazon -lr 0.05 -dim 200 -minCount 2 -epoch 20 -minn 3 -maxn 6
fasttext cbow -input raw_data/amazon_year.txt -output ./fasttext_cbow/amazon/year/amazon -lr 0.05 -dim 200 -minCount 2 -epoch 20 -minn 3 -maxn 6
fasttext cbow -input raw_data/amazon_month.txt -output ./fasttext_cbow/amazon/month/amazon -lr 0.05 -dim 200 -minCount 2 -epoch 20 -minn 3 -maxn 6

# yelp restaurant
fasttext cbow -input raw_data/yelp_rest.txt -output ./fasttext_cbow/yelp/rest/yelp_rest -lr 0.05 -dim 200 -minCount 2 -epoch 20 -minn 3 -maxn 6
fasttext cbow -input raw_data/yelp_rest_year.txt -output ./fasttext_cbow/yelp/rest/year/yelp_rest -lr 0.05 -dim 200 -minCount 2 -epoch 20 -minn 3 -maxn 6
fasttext cbow -input raw_data/yelp_rest_month.txt -output ./fasttext_cbow/yelp/rest/month/yelp_rest -lr 0.05 -dim 200 -minCount 2 -epoch 20 -minn 3 -maxn 6

# yelp hotel data
fasttext cbow -input raw_data/yelp_hotel.txt -output ./fasttext_cbow/yelp/hotel/yelp_hotel -lr 0.05 -dim 200 -minCount 2 -epoch 20 -minn 3 -maxn 6
fasttext cbow -input raw_data/yelp_hotel_year.txt -output ./fasttext_cbow/yelp/hotel/year/yelp_hotel -lr 0.05 -dim 200 -minCount 2 -epoch 20 -minn 3 -maxn 6
fasttext cbow -input raw_data/yelp_hotel_month.txt -output ./fasttext_cbow/yelp/hotel/month/yelp_hotel -lr 0.05 -dim 200 -minCount 2 -epoch 20 -minn 3 -maxn 6

# vaccine
fasttext cbow -input raw_data/vaccine.txt -output ./fasttext_cbow/vaccine/vaccine -lr 0.05 -dim 200 -minCount 2 -epoch 20 -minn 3 -maxn 6
fasttext cbow -input raw_data/vaccine_year.txt -output ./fasttext_cbow/vaccine/year/vaccine -lr 0.05 -dim 200 -minCount 2 -epoch 20 -minn 3 -maxn 6
fasttext cbow -input raw_data/vaccine_month.txt -output ./fasttext_cbow/vaccine/month/vaccine -lr 0.05 -dim 200 -minCount 2 -epoch 20 -minn 3 -maxn 6

# dianping
fasttext cbow -input raw_data/dianping.txt -output ./fasttext_cbow/dianping/dianping -lr 0.05 -dim 200 -minCount 2 -epoch 20 -minn 3 -maxn 6
fasttext cbow -input raw_data/dianping_year.txt -output ./fasttext_cbow/dianping/year/dianping -lr 0.05 -dim 200 -minCount 2 -epoch 20 -minn 3 -maxn 6
fasttext cbow -input raw_data/dianping_month.txt -output ./fasttext_cbow/dianping/month/dianping -lr 0.05 -dim 200 -minCount 2 -epoch 20 -minn 3 -maxn 6

# economy
fasttext cbow -input raw_data/economy.txt -output ./fasttext_cbow/economy/economy -lr 0.05 -dim 200 -minCount 2 -epoch 20 -minn 3 -maxn 6
fasttext cbow -input raw_data/economy_year.txt -output ./fasttext_cbow/economy/year/economy -lr 0.05 -dim 200 -minCount 2 -epoch 20 -minn 3 -maxn 6
fasttext cbow -input raw_data/economy_month.txt -output ./fasttext_cbow/economy/month/economy -lr 0.05 -dim 200 -minCount 2 -epoch 20 -minn 3 -maxn 6

# parties, this data is from our ACL'18 paper, 
# Xiaolei Huang and Michael J. Paul. Examining temporality in document classification. Association for Computational Linguistics (ACL), Melbourne, Australia. July 2018.
#fasttext cbow -input raw_data/parties_year.txt -output ./fasttext_cbow/parties/year/parties -lr 0.05 -dim 200 -minCount 2 -epoch 20 -minn 3 -maxn 6
