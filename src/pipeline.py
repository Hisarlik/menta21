from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import string

punt_vocab = string.punctuation


def define_pipeline():
    estimators = [  ('tfidf_ngrams', TfidfVectorizer(analyzer='char_wb', 
                                                    min_df=0.1, 
                                                    ngram_range=(1, 6))), 
                    # ('tfidf_punctuation', TfidfVectorizer(analyzer='char_wb', 
                    #                                       vocabulary=punt_vocab, 
                    #                                       ngram_range=(1, 6))) 
                    ('scaler', StandardScaler(with_mean=False)),

                ]


    pipe = FeatureUnion(estimators)
    return pipe

