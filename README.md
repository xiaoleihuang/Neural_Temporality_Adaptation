# Neural Temporality Adaptation and Diachronic (Dynamic) Word Embeddings
Source codes for our paper "[Neural Temporality Adaptation for Document Classification: Diachronic Word Embeddings and Domain Adaptation Models](https://cmci.colorado.edu/~mpaul/files/acl2019_temporality.pdf)" at ACL 2019.

## Word Semantic Meanings Shift Overtime
![Image of Semantic Shifts](https://nlp.stanford.edu/projects/histwords/images/wordpaths.png)
> Image is from https://nlp.stanford.edu/projects/histwords/

The meanings of words shift overtime~ This will bring tremendous effects on document models.

## How Shifts Impact Document Classifiers
![Impacts of the Shifts on Document Classifiers](https://github.com/xiaoleihuang/Neural_Temporality_Adaptation/blob/master/git_images/impacts.png)

Overall, it is clear that classifiers generally perform best when applied to the same time interval they were trained. Performance diminishes when applied to different time intervals, although different corpora exhibit differ patterns in the way in which the performance diminishes.

## Language Odyssey, What & Why the shifts happen?
![Image of Semantic Shifts](https://github.com/xiaoleihuang/Neural_Temporality_Adaptation/blob/master/git_images/shifts.png)

We explore and analyze the shifts from three perspectives:
1. Word usage change: the way to express opinions change;
2. Word context change: contextual words are important parts to train word embeddings. The change of contextual words impact word embeddings and therefore neural document classifiers based on word embeddings.
3. Semantic distance: after obtaining diachronic word embeddings, we treat each time period as a domain and then use [Wasserstein distance]() to measure time shifts.

Generally, closer time intervals share higher overlap and have smaller semantic distance shifts, and vice versa.

## Test Platform
Python 3.6+, Ubuntu 16.04

## Experiment Preparation
  1. Install [Conda](https://www.anaconda.com/distribution/) and then install required Python packages via `pip install -r requirements.txt`;
  2. Train and obtain regular and diachronic word embeddings. Please refer to readme.md in the `embeddings` folder;
  3. Create domain data: `python create_domain_data.py`;
  4. Create general and domain tokenizers: `python create_domain_tokenizer.py`;
  5. Create weights for the embedding layer: `python create_weights.py`;
  6. Create train/valid/test data: `python create_split_data.py`;
  7. Convert data into indices: `python create_word2idx.py`.

## Data Analysis
  1. Please refer to the `analysis` folder.
  2. There are three main analysis perspectives: word usage, word context and semantic distance.

## Baselines
  1. Please refer to the readme.md in the `baselines` folder.
  2. The datasets of baselines will be saved in the `baselines_data` folder.

## Our approach
