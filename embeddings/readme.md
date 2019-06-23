This is the repository to train diachronic word embeddings.
Send me an email if you have any questions, [xiaolei.huang@colorado.edu](mailto:xiaolei.huang@colorado.edu).

## Steps
1. We provide [whole resources](https://drive.google.com/open?id=1Z4j7WhBhe8Df-0PVZI0vKj4vhU9rAo6T) for the following two steps via Google Drive.
2. Obtaining the corpora (corpora.zip) to train diachronic word embeddings (optional).
  * Download raw and our processed datasets.
  * To train the diachronic word embeddings, two steps:
    * Step 1: Preprocessed Data (optional, because the data was provided in the zipped corpora)
    * Step 2: Install [Facebook FastText](https://github.com/facebookresearch/fastText#building-fasttext-using-cmake) and run the shell code: `sh train_fasttext_cbow.sh`. You can change to the skip-gram mode and change parameters. **Don't forget to change data path in the shell script.**
3. Download pretrained regular embeddings and diachronic word embeddings (diachronic_word_embeddings.zip).
