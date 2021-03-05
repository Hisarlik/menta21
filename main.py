import pandas as pd
from src.pipeline import define_pipeline
import numpy as np
from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score

if __name__ == "__main__":
    
    print("Start experiment")

    train = "data/temp/train_2000.csv"
    dev = "data/temp/dev_2000.csv"

    train_df = pd.read_csv(train)
    dev_df = pd.read_csv(dev)

    print(train_df.head())

    texts = np.hstack([train_df['text1'].values, train_df['text2'].values])
    texts1 = train_df['text1']
    texts2 = train_df['text2']
    Y_train = train_df['same'].values

    pipe = define_pipeline()

    pipe.fit(texts)

    x1 = pipe.transform(texts1)
    x2 = pipe.transform(texts2)
    X_train = np.abs(x1-x2).todense()

    # clf = LogisticRegression(solver='lbfgs', max_iter=500)
    # distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2'])
    # param_clf = RandomizedSearchCV(clf, distributions, random_state=0, verbose=2, scoring='roc_auc')
    # search = param_clf.fit(X_train, Y_train)
    # print(search.best_params_)


    clf = LogisticRegression(C=0.226, solver='lbfgs', max_iter=5000, verbose=True)
    clf.fit(X_train, Y_train)

    texts = np.hstack([dev_df['text1'].values, dev_df['text2'].values])
    texts1 = dev_df['text1']
    texts2 = dev_df['text2']
    Y_dev = dev_df['same'].values

    x1 = pipe.transform(texts1)
    x2 = pipe.transform(texts2)
    X_dev = np.abs(x1-x2).todense()

    preds = clf.predict(X_dev)
    print("f1_score:",f1_score(Y_dev, preds, average='macro'))


    threshold = 0.65
    preds = (clf.predict_proba(X_dev)[:,1] >= threshold).astype(bool)
    print("f1_score:",f1_score(Y_dev, preds, average='macro'))

    