import joblib
import random
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelEncoder



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

def vectorize_dataset(config):

    conf = {}

    # Deterministic
    random.seed(hash("setting random seeds") % 2**30 - 1)
    np.random.seed(hash("improves reproducibility") % 2**30 - 1)


    path = config['path_dataset']
    train_data = path+"train.csv"
    dev_data = path+"dev.csv"
    test_data = path+"test.csv"

    print("Start vect")

    train_df = pd.read_csv(train_data)

    texts1 = train_df['text1']
    texts2 = train_df['text2']
    train_df['same'] = LabelEncoder().fit_transform(train_df["same"])
    Y_train = train_df['same'].values
    train_df = None

    Y_train_memmap = np.memmap(path + 'Y_train.npy', dtype='int32', mode='w+', shape=(len(Y_train)))
    Y_train_memmap[:] = Y_train
    print(Y_train_memmap)
    Y_train_memmap.flush()
    Y_train_memmap = None

    pipe_ngrams = define_vectorizer("ngrams")
    pipe_ngrams.fit(texts1[:15000])
    transformer_size = len(pipe_ngrams.named_steps['tfidf_ngrams'].get_feature_names())
    X_train = np.memmap(path + 'features_ngrams_X_train.npy', dtype='float32', mode='w+', shape=(len(texts1), transformer_size))
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

    X_train_ngrams = np.memmap(path + 'features_ngrams_X_train.npy', dtype='float32', mode='r', shape=(len(texts1), transformer_size))
    print(X_train_ngrams.shape)
    X_train_ngrams = None
    X_train = None

    pipe_punct = define_vectorizer("punct")
    pipe_punct.fit(texts1[:15000])
    transformer_size = len(pipe_punct.named_steps['tfidf_punctuation'].get_feature_names())
    X_train = np.memmap(path + 'features_punct_X_train.npy', dtype='float32', mode='w+', shape=(len(texts1), transformer_size))
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

    Y_test_memmap = np.memmap(path + 'Y_dev.npy', dtype='int32', mode='w+', shape=(len(Y_test)))
    Y_test_memmap[:] = Y_test
    print(Y_test_memmap)
    Y_test_memmap.flush()
    Y_test_memmap = None

    transformer_size = len(pipe_ngrams.named_steps['tfidf_ngrams'].get_feature_names())
    X_test = np.memmap(path + 'features_ngrams_X_dev.npy', dtype='float32', mode='w+', shape=(len(texts1), transformer_size))
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
    X_test = np.memmap(path + 'features_punct_X_dev.npy', dtype='float32', mode='w+', shape=(len(texts1), transformer_size))

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
    Y_test_memmap = np.memmap(path+'Y_test.npy', dtype='int32', mode='w+', shape=(len(Y_test)))
    Y_test_memmap[:] = Y_test
    Y_test_memmap.flush()
    transformer_size = len(pipe_ngrams.named_steps['tfidf_ngrams'].get_feature_names())

    X_test = np.memmap(path+'features_ngrams_X_test.npy', dtype='float32', mode='w+', shape=(len(texts1), transformer_size))
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
    X_test = np.memmap(path+'features_punct_X_test.npy', dtype='float32', mode='w+', shape=(len(texts1), transformer_size))
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

    joblib.dump(pipe_ngrams, path+'pipe_ngrams.pkl')
    joblib.dump(pipe_punct, path+'pipe_punct.pkl')  
    joblib.dump(conf, path+'conf.pkl')  

    c = joblib.load(path+'conf.pkl')
    print(c)