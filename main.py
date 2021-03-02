import pandas as pd
from src.pipeline import define_pipeline
import numpy as np

if __name__ == "__main__":
    
    print("Start experiment")

    train = "data/temp/train_tiny.csv"
    dev = "data/temp/dev_tiny.csv"

    train_df = pd.read_csv(train)
    dev_df = pd.read_csv(dev)

    print(train_df.head())

    texts = np.hstack([train_df['text1'].values, train_df['text2'].values])
    texts1 = train_df['text1']
    texts2 = train_df['text2']
    Y_train = train_df['same'].values

    pipe = define_pipeline()

    x1 = pipe.fit(texts1)
    x2 = pipe.fit(texts2)
    X_train = np.abs(x1-x2).todense()

    clf = LogisticRegression(solver='lbfgs', max_iter=500)
    distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2'])
    param_clf = RandomizedSearchCV(clf, distributions, random_state=0, verbose=2, scoring='roc_auc')
    search = param_clf.fit(X_train, Y_train)
    print(search.best_params_)

    