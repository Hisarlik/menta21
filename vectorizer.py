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

def fit_vectorizer(config):

    limit_vectorizer = config['limit_data_vectorizer']

    path = config['path_dataset']
    train_data = path+"train.csv"


    ## Load data
    data_df = pd.read_csv(train_data)   
    if len(data_df)>limit_vectorizer:
        data_df = data_df[:limit_vectorizer]
    texts = data_df['text1'] 

    print("Data Loaded for vectorization:")
    print("ngrams and  punctuation vectorizer:", len(texts))
    print("labels vectorizer:", data_df["same"].value_counts()) 

    ## labels
    label_encoder = LabelEncoder().fit(data_df["same"])
    
    
    print("define ngrams")
    ## ngrams vect
    pipe_ngrams = define_vectorizer("ngrams")
    pipe_ngrams.fit(texts)
    
    print("define punct")
    ##punct vect
    pipe_punct = define_vectorizer("punct")
    pipe_punct.fit(texts)
    
    
    return pipe_ngrams, pipe_punct, label_encoder

def vectorize_dataset(config):

    conf = {}

    # Deterministic
    random.seed(hash("setting random seeds") % 2**30 - 1)
    np.random.seed(hash("improves reproducibility") % 2**30 - 1)


    path = config['path_dataset']
    train_data = path+"train.csv"


    ############# fit ###################################

    print("Start vectorization")
    pipe_ngrams, pipe_punct, label_encoder =  fit_vectorizer(config)

    ############### train #####################################
    print("Loading train data")

    print("Transform labels")
    train_df = pd.read_csv(train_data)
    train_length = len(train_df['text1'])
    Y_train = label_encoder.transform(train_df["same"].values)
    train_df = None
    Y_train_memmap = np.memmap(path + 'Y_train.npy', dtype='int32', mode='w+', shape=(len(Y_train)))
    Y_train_memmap[:] = Y_train
    Y_train_memmap.flush()
    Y_train_memmap = None

    print("Transform ngrams")
    transformer_size = len(pipe_ngrams.named_steps['tfidf_ngrams'].get_feature_names())
    X_train = np.memmap(path + 'features_ngrams_X_train.npy', dtype='float32', mode='w+', shape=(train_length, transformer_size))
    chunksize = 1000
    reader = pd.read_csv(train_data, iterator=True, chunksize=chunksize)
    i = 0
    for chunk in reader:
        size = len(chunk)
        print(len(chunk))
        id_low = chunksize*i
        id_high =  id_low +size
        x1 = pipe_ngrams.transform(chunk['text1'])
        x2 = pipe_ngrams.transform(chunk['text2'])
        X_train[id_low:id_high] = np.abs(x1-x2).todense()
        i+= 1
    print("Shape ngrams transformed data:", X_train.shape)
    conf['rows_train'] = X_train.shape[0]
    conf['ngrams'] = X_train.shape[1]
    X_train.flush()
    X_train = None

    print("Transform punctuation")
    transformer_size = len(pipe_punct.named_steps['tfidf_punctuation'].get_feature_names())
    X_train = np.memmap(path + 'features_punct_X_train.npy', dtype='float32', mode='w+', shape=(train_length, transformer_size))
    reader = pd.read_csv(train_data, iterator=True, chunksize=chunksize)
    i = 0
    for chunk in reader:
        size = len(chunk)
        print(len(chunk))
        id_low = chunksize*i
        id_high =  id_low +size
        x1 = pipe_punct.transform(chunk['text1'])
        x2 = pipe_punct.transform(chunk['text2'])
        X_train[id_low:id_high] = np.abs(x1-x2).todense()
        i+= 1
    print("Shape punctuation transformed data:",X_train.shape)
    conf['punct'] = X_train.shape[1]
    X_train.flush()
    X_train = None


    ##### store 

    joblib.dump(pipe_ngrams, path+'pipe_ngrams.pkl')
    joblib.dump(pipe_punct, path+'pipe_punct.pkl')  
    joblib.dump(label_encoder, path+'label_encoder.pkl') 
    joblib.dump(conf, path+'conf.pkl')  

def vectorize_predict(config):

    conf = {}

    path_model = config['path_model']
    path_predict = config['path_predict']
    path_predict_data = path_model+"temp.csv"
 
    pipe_ngrams = joblib.load( path_model+'pipe_ngrams.pkl')
    pipe_punct = joblib.load(path_model+'pipe_punct.pkl')  
    print("vectorizers loaded")

    ############### test #####################################
    print("Loading predict data")
    predict_df = pd.read_csv(path_predict_data)
    predict_length = len(predict_df['text1'])
    print("Number of predictions",predict_length)

    print("Transform ngrams")
    transformer_size = len(pipe_ngrams.named_steps['tfidf_ngrams'].get_feature_names())
    X_predict = np.memmap(path_predict + 'features_ngrams_X_predict.npy', dtype='float32', mode='w+', shape=(predict_length, transformer_size))
    chunksize = 1000
    reader = pd.read_csv(path_predict_data, iterator=True, chunksize=chunksize)
    i = 0
    for chunk in reader:
        size = len(chunk)
        id_low = chunksize*i
        id_high =  id_low +size
        x1 = pipe_ngrams.transform(chunk['text1'])
        x2 = pipe_ngrams.transform(chunk['text2'])
        X_predict[id_low:id_high] = np.abs(x1-x2).todense()
        i+= 1
    print("Shape ngrams transformed data:",X_predict.shape)
    conf['rows_predict'] = X_predict.shape[0]
    conf['ngrams'] = X_predict.shape[1]
    X_predict.flush()
    X_predict = None

    print("Transform punctuation")
    transformer_size = len(pipe_punct.named_steps['tfidf_punctuation'].get_feature_names())
    X_predict = np.memmap(path_predict + 'features_punct_X_predict.npy', dtype='float32', mode='w+', shape=(predict_length, transformer_size))
    reader = pd.read_csv(path_predict_data, iterator=True, chunksize=chunksize)
    i = 0
    for chunk in reader:
        size = len(chunk)
        id_low = chunksize*i
        id_high =  id_low +size
        x1 = pipe_punct.transform(chunk['text1'])
        x2 = pipe_punct.transform(chunk['text2'])
        X_predict[id_low:id_high] = np.abs(x1-x2).todense()
        i+= 1
    print("Shape punctuation transformed data:",X_predict.shape)
    conf['punct'] = X_predict.shape[1]
    X_predict.flush()
    X_predict = None

    joblib.dump(conf, path_predict+'conf_predict.pkl')

