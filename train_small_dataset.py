import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
import string
import random
import scipy
from scipy.stats import uniform
import joblib


from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelEncoder

from model import model_pipeline

sklearn_random = 20


path_training_small_truth = "data/pan20-authorship-verification-training-small-truth.jsonl"
path_training_small = "data/pan20-authorship-verification-training-small.jsonl"

path_dataset = "data/small/"

def create_dataset():
    df_texts = pd.read_json(path_training_small, lines=True)
    df_truth = pd.read_json(path_training_small_truth, lines=True)

    df_join_training_data = pd.concat([df_truth, df_texts], axis=1).reindex(df_truth.index)
    df_join_training_data = df_join_training_data.loc[:,~df_join_training_data.columns.duplicated()]

    df_join_training_data[['text1','text2']] = pd.DataFrame(df_join_training_data.pair.tolist(), index= df_join_training_data.index)
    df_join_training_data[['author1','author2']] = pd.DataFrame(df_join_training_data.authors.tolist(), index= df_join_training_data.index)

    df_join_training_data = df_join_training_data.drop(columns=["pair", "fandoms","authors"])

    dataset = df_join_training_data.sample(frac=1).reset_index(drop=True)

    print(dataset.head())

    train, test = train_test_split(dataset, test_size=0.15, random_state=sklearn_random)

    train, dev = train_test_split(train, test_size=0.20, random_state=sklearn_random)

    train.to_csv(path_dataset+"train_complete.csv", index=False)
    dev.to_csv(path_dataset+"dev_complete.csv", index=False)
    test.to_csv(path_dataset+"test_complete.csv", index=False)

def define_vectorizer(type):

    if type == "ngrams":
      estimators = [  
                      ('tfidf_ngrams', TfidfVectorizer(analyzer='char_wb',norm='l1', min_df=0.05, ngram_range=(1, 6))),
                      ('scaler', StandardScaler(with_mean=False)),

                  ]
    elif type == "punct":
      punt_vocab = string.punctuation
      estimators = [  
                      ('tfidf_punctuation', TfidfVectorizer(analyzer='char_wb', 
                                                            vocabulary=punt_vocab, 
                                                            ngram_range=(1, 6))), 
                      ('scaler', StandardScaler(with_mean=False)),

                  ]


    pipe = Pipeline(estimators)
    return pipe




def vectorizer():

    conf = {}

    # Deterministic
    random.seed(hash("setting random seeds") % 2**30 - 1)
    np.random.seed(hash("improves reproducibility") % 2**30 - 1)

    config = dict(
        epochs = 50,
        batch_size = 10,
        learning_rate = 0.001,
        dataset = "Authorship 2000",
        architecture = "Dense:  Input, Layer 512, relu, batchnorm 512 , Layer 64, relu, batchnorm 64, dropout 0.1, output", 
        criterion = "BCEWithLogitsLoss",
        optimizer = "Adam"

    )


    print("Start experiment")

    train_data = path_dataset+"train_complete.csv"
    dev_data = path_dataset+"dev_complete.csv"
    test_data = path_dataset+"test_complete.csv"

    train_df = pd.read_csv(train_data)

    texts1 = train_df['text1']
    texts2 = train_df['text2']
    train_df['same'] = LabelEncoder().fit_transform(train_df["same"])
    Y_train = train_df['same'].values
    train_df = None

    Y_train_memmap = np.memmap(path_dataset + 'Y_train.npy', dtype='int32', mode='w+', shape=(len(Y_train)))
    Y_train_memmap[:] = Y_train
    print(Y_train_memmap)
    Y_train_memmap.flush()
    Y_train_memmap = None

    pipe_ngrams = define_vectorizer("ngrams")
    pipe_ngrams.fit(texts1[:15000])
    transformer_size = len(pipe_ngrams.named_steps['tfidf_ngrams'].get_feature_names())
    X_train = np.memmap(path_dataset + 'features_ngrams_X_train.npy', dtype='float32', mode='w+', shape=(len(texts1), transformer_size))
    print(X_train.shape)
    chunksize = 1000
    reader = pd.read_csv(train_data, iterator=True, chunksize=chunksize)
    i = 0
    for chunk in reader:
        size = len(chunk)
        print(len(chunk))
        id_low = chunksize*i
        id_high =  id_low +size
        print(f"Iterator: [{id_low},{id_high}]")
        x1 = pipe_ngrams.transform(chunk['text1'])
        x2 = pipe_ngrams.transform(chunk['text2'])
        X_train[id_low:id_high] = np.abs(x1-x2).todense()
        i+= 1
    print(X_train.shape)
    conf['rows_train'] = X_train.shape[0]
    conf['ngrams'] = X_train.shape[1]
    X_train.flush()

    X_train_ngrams = np.memmap(path_dataset + 'features_ngrams_X_train.npy', dtype='float32', mode='r', shape=(len(texts1), transformer_size))
    print(X_train_ngrams.shape)
    X_train_ngrams = None
    X_train = None

    pipe_punct = define_vectorizer("punct")
    pipe_punct.fit(texts1[:15000])
    transformer_size = len(pipe_punct.named_steps['tfidf_punctuation'].get_feature_names())
    X_train = np.memmap(path_dataset + 'features_punct_X_train.npy', dtype='float32', mode='w+', shape=(len(texts1), transformer_size))
    reader = pd.read_csv(train_data, iterator=True, chunksize=chunksize)
    i = 0
    for chunk in reader:
        size = len(chunk)
        print(len(chunk))
        id_low = chunksize*i
        id_high =  id_low +size
        print(f"Iterator: [{id_low},{id_high}]")
        x1 = pipe_punct.transform(chunk['text1'])
        x2 = pipe_punct.transform(chunk['text2'])
        X_train[id_low:id_high] = np.abs(x1-x2).todense()
        i+= 1
    print(X_train.shape)
    conf['punct'] = X_train.shape[1]

    ##dev_df
    dev_df = pd.read_csv(dev_data)
    texts = np.hstack([dev_df['text1'].values, dev_df['text2'].values])
    texts1 = dev_df['text1']
    texts2 = dev_df['text2']
    dev_df['same'] = LabelEncoder().fit_transform(dev_df["same"])
    Y_test = dev_df['same'].values

    Y_test_memmap = np.memmap(path_dataset + 'Y_dev.npy', dtype='int32', mode='w+', shape=(len(Y_test)))
    Y_test_memmap[:] = Y_test
    print(Y_test_memmap)
    Y_test_memmap.flush()
    Y_test_memmap = None

    transformer_size = len(pipe_ngrams.named_steps['tfidf_ngrams'].get_feature_names())
    X_test = np.memmap(path_dataset + 'features_ngrams_X_dev.npy', dtype='float32', mode='w+', shape=(len(texts1), transformer_size))
    reader = pd.read_csv(dev_data, iterator=True, chunksize=chunksize)
    print(X_test.shape)
    conf['rows_dev'] = X_test.shape[0]

    i = 0
    for chunk in reader:
        size = len(chunk)
        print(len(chunk))
        id_low = chunksize*i
        id_high =  id_low +size
        print(f"Iterator: [{id_low},{id_high}]")
        x1 = pipe_ngrams.transform(chunk['text1'])
        x2 = pipe_ngrams.transform(chunk['text2'])
        X_test[id_low:id_high] = np.abs(x1-x2).todense()
        i+= 1
    transformer_size = len(pipe_punct.named_steps['tfidf_punctuation'].get_feature_names())
    X_test = np.memmap(path_dataset + 'features_punct_X_dev.npy', dtype='float32', mode='w+', shape=(len(texts1), transformer_size))

    reader = pd.read_csv(dev_data, iterator=True, chunksize=chunksize)
    i = 0
    for chunk in reader:
        size = len(chunk)
        print(len(chunk))
        id_low = chunksize*i
        id_high =  id_low +size
        print(f"Iterator: [{id_low},{id_high}]")
        x1 = pipe_punct.transform(chunk['text1'])
        x2 = pipe_punct.transform(chunk['text2'])
        X_test[id_low:id_high] = np.abs(x1-x2).todense()
        i+= 1



    dev_df = pd.read_csv(test_data)
    texts = np.hstack([dev_df['text1'].values, dev_df['text2'].values])
    texts1 = dev_df['text1']
    texts2 = dev_df['text2']
    dev_df['same'] = LabelEncoder().fit_transform(dev_df["same"])
    Y_test = dev_df['same'].values
    Y_test_memmap = np.memmap(path_dataset+'Y_test.npy', dtype='int32', mode='w+', shape=(len(Y_test)))
    Y_test_memmap[:] = Y_test
    Y_test_memmap.flush()
    transformer_size = len(pipe_ngrams.named_steps['tfidf_ngrams'].get_feature_names())

    X_test = np.memmap(path_dataset+'features_ngrams_X_test.npy', dtype='float32', mode='w+', shape=(len(texts1), transformer_size))
    print(X_test.shape)
    conf['rows_test'] = X_test.shape[0]
    reader = pd.read_csv(test_data, iterator=True, chunksize=chunksize)
    i = 0
    for chunk in reader:
        size = len(chunk)
        print(len(chunk))
        id_low = chunksize*i
        id_high =  id_low +size
        print(f"Iterator: [{id_low},{id_high}]")
        x1 = pipe_ngrams.transform(chunk['text1'])
        x2 = pipe_ngrams.transform(chunk['text2'])
        X_test[id_low:id_high] = np.abs(x1-x2).todense()
        i+= 1
    transformer_size = len(pipe_punct.named_steps['tfidf_punctuation'].get_feature_names())
    X_test = np.memmap(path_dataset+'features_punct_X_test.npy', dtype='float32', mode='w+', shape=(len(texts1), transformer_size))
    reader = pd.read_csv(test_data, iterator=True, chunksize=chunksize)

    i = 0
    for chunk in reader:
        size = len(chunk)
        print(len(chunk))
        id_low = chunksize*i
        id_high =  id_low +size
        print(f"Iterator: [{id_low},{id_high}]")
        x1 = pipe_punct.transform(chunk['text1'])
        x2 = pipe_punct.transform(chunk['text2'])
        X_test[id_low:id_high] = np.abs(x1-x2).todense()
        i+= 1

    joblib.dump(pipe_ngrams, path_dataset+'pipe_ngrams.pkl')
    joblib.dump(pipe_punct, path_dataset+'pipe_punct.pkl')  
    joblib.dump(conf, path_dataset+'conf.pkl')  





if __name__ == "__main__":

    config = dict(
        epochs = 10,
        batch_size = 80,
        learning_rate = 0.001,
        dataset = "Authorship 2000",
        architecture = "Dense:  Input, Layer 512, relu, batchnorm 512 , Layer 64, relu, batchnorm 64, dropout 0.1, output", 
        criterion = "BCEWithLogitsLoss",
        optimizer = "Adam"
    )



    #create_dataset()
    #vectorizer()
    model_pipeline(config)

    #train = pd.read_csv(path_dataset+"train_complete.csv")
    #train = train[:500]

    #dev = pd.read_csv(path_dataset+"dev_complete.csv")
    #dev = train[:500]

    #test = pd.read_csv(path_dataset+"test_complete.csv")
    #test = train[:500]


    #train.to_csv(path_dataset+"train_complete.csv", index=False)
    #dev.to_csv(path_dataset+"dev_complete.csv", index=False)
    #test.to_csv(path_dataset+"test_complete.csv", index=False)


    c = joblib.load(path_dataset+'conf.pkl')
    print(c)