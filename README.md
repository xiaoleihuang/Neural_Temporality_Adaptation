# Neural Temporality Adaptation and Diachronic (Dynamic) Word Embeddings
Source codes for our paper "[Neural Temporality Adaptation for Document Classification: Diachronic Word Embeddings and Domain Adaptation Models](https://cmci.colorado.edu/~mpaul/files/acl2019_temporality.pdf)" at ACL 2019.

## Word Semantic Meanings Shift Overtime
![Image of Semantic Shifts](https://nlp.stanford.edu/projects/histwords/images/wordpaths.png)
> Image is from https://nlp.stanford.edu/projects/histwords/

The meanings of words shift overtime~ This will bring tremendous effects on document models.

## How Shifts Impact Document Classifiers
![Impacts of the Shifts on Document Classifiers](https://github.com/xiaoleihuang/Neural_Temporality_Adaptation/blob/master/git_images/impacts.png)

Overall, it is clear that classifiers generally perform best when applied to the same time interval they were trained. Performance diminishes when applied to different time intervals, although different corpora exhibit differ patterns in the way in which the performance diminishes. The image is from our previous publication, [Examining Temporality in Document Classification](https://aclweb.org/anthology/papers/P/P18/P18-2110/).

## Language Odyssey, What & Why the shifts happen?
![Image of Semantic Shifts](https://github.com/xiaoleihuang/Neural_Temporality_Adaptation/blob/master/git_images/shifts.png)

We explore and analyze the shifts from three perspectives:
1. Word usage change: the way to express opinions change;
2. Word context change: contextual words are important parts to train word embeddings. The change of contextual words impact word embeddings and therefore neural document classifiers based on word embeddings.
3. Semantic distance: after obtaining diachronic word embeddings, we treat each time period as a domain and then use [Wasserstein distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html) to measure time shifts.

Generally, closer time intervals share higher overlap and have smaller semantic distance shifts, and vice versa.

## Test Platform
Python 3.6+, Ubuntu 16.04

## Experiment Preparation
  1. Install [Conda](https://www.anaconda.com/distribution/) and then install required Python packages via `pip install -r requirements.txt`;
  2. Train and obtain regular and diachronic word embeddings. Please refer to readme.md in the `embeddings` folder;
  3. Create domain data: `python create_domain_data.py`;
  4. Create general and domain tokenizers: `python create_domain_tokenizer.py`;
  5. Create weights for the embedding layer: `python create_weights.py`;
  6. Create train/valid/test data: You can either download [our processed split data](https://drive.google.com/open?id=1JakAO-sN-VfR4UY5XFqu3Dbs8x-1fN6E) or run `python create_split_data.py`;
  7. Convert data into indices: `python create_word2idx.py`.

## Data Analysis
  1. Please refer to the `analysis` folder.
  2. There are three main analysis perspectives: word usage, word context and semantic distance.

## Baselines
  1. Please refer to the readme.md in the `baselines` folder.
  2. The datasets of baselines will be saved in the `baselines_data` folder.

## Our approach

## Contacts
Because the experimental datasets are too large to share all of them. Please send any requests or questions to my email: [xiaolei.huang@colorado.edu](xiaolei.huang@colorado.edu).

## Citation
Please consider cite our work as the following:
```
@inproceedings{huang-paul-2019-diachronic,
    title = "Neural Temporality Adaptation for Document Classification: Diachronic Word Embeddings and Domain Adaptation Models",
    author = "Huang, Xiaolei and Paul, Michael J.",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://cmci.colorado.edu/~mpaul/files/acl2019_temporality.pdf",
    pages = "136--146",
    abstract = "Language usage can change across periods of time, but document classifiers models are usually trained and tested on corpora spanning multiple years without considering temporal variations. This paper describes two complementary ways to adapt classifiers to shifts across time. First, we show that diachronic word embeddings, which were originally developed to study language change, can also improve document classification, and we show a simple method for constructing this type of embedding. Second, we propose a time-driven neural classification model inspired by methods for domain adaptation. Experiments on six corpora show how these methods can make classifiers more robust over time.",
}
```
