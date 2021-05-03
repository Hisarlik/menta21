import os
import pandas as pd
import wandb
from model_large_dataset import model_pipeline, predict_model
from vectorizers_large_dataset import vectorize_dataset, vectorize_predict
from sklearn.model_selection import train_test_split



sklearn_random = 20

def store_pandas(dataframe, path):

    # if file does not exist write header 
    if not os.path.isfile(path):
       dataframe.to_csv(path, index=False)
    else: # else it exists so append without writing the header
       dataframe.to_csv(path, mode='a', header=False, index=False)


def create_dataset(config):
    
    ## load data
    df_texts = pd.read_json(config['path_training'], lines=True, chunksize=1000)
    df_truth = pd.read_json(config['path_training_truth'], lines=True, chunksize=1000)
    print("data loaded")

    path = config['path_dataset']+"train.csv"
    try:
        os.stat(config['path_dataset'])
    except:
        os.mkdir(config['path_dataset'])

    ## create temp file
    for texts, truth in zip(df_texts, df_truth):
        df_join_training_data = pd.concat([truth, texts], axis=1).reindex(truth.index)
        df_join_training_data = df_join_training_data.loc[:,~df_join_training_data.columns.duplicated()]
        df_join_training_data[['text1','text2']] = pd.DataFrame(df_join_training_data.pair.tolist(), index= df_join_training_data.index)
        df_join_training_data = df_join_training_data.drop(columns=["pair", "fandoms","authors"])

        store_pandas(df_join_training_data, path)
    
    df_texts=None
    df_truth=None
    print("temp csv stored")

    ## randomize data
    randomize_dataset(config)
    print("csv randomized")


def parse_predict_data(config):
    df_texts = pd.read_json(config['path_predict'], lines=True, chunksize=1000)      
    print("data loaded") 
    path = config['path_dataset']+"predict.csv"

    for chunk in df_texts:
        chunk[['text1','text2']] = pd.DataFrame(chunk.pair.tolist(), index= chunk.index)
        chunk = chunk.drop(columns=["pair", "fandoms"])

        store_pandas(chunk, path)
    
    print("temp csv stored")  
    df_texts=None
    df_truth=None



def randomize_dataset(config):

    path_train = config['path_dataset']+"train.csv"
    path_dev = config['path_dataset']+"dev.csv"
    path_test = config['path_dataset']+"test.csv"

    dataset = pd.read_csv(path_train)

    dataset = dataset.sample(frac=1).reset_index(drop=True)

    limit_dataset = config.get("limit_dataset")
    if limit_dataset:
        dataset = dataset[:limit_dataset]

    dataset, test = train_test_split(dataset, test_size=0.15, random_state=sklearn_random)
    dataset, dev = train_test_split(dataset, test_size=0.18, random_state=sklearn_random)
    dataset.to_csv(path_train, index=False)
    dev.to_csv(path_dev, index=False)
    test.to_csv(path_test, index=False)

def train_model(epochs, batch_size, learning_rate=0.001):

    config = dict(
        path_training="data/pan20-authorship-verification-training-large.jsonl",
        path_training_truth="data/pan20-authorship-verification-training-large-truth.jsonl",     
        epochs = epochs,
        batch_size = batch_size,
        learning_rate = learning_rate,
        dataset = "Authorship 2000",
        architecture = "Dense:  Input, Layer 512, relu, batchnorm 512 , Layer 64, relu, batchnorm 64, dropout 0.1, output", 
        criterion = "BCEWithLogitsLoss",
        optimizer = "Adam",
        limit_dataset = None,
        path_dataset = "data/large/",
        limit_data_vectorizer = 15000
        
    )


    result = model_pipeline(config)





if __name__ == "__main__":

    # Wandb Login
    wandb.login()

    #create_dataset(config)
    #vectorize_dataset(config)



    train_model(1, 368, 0.00005)
    train_model(5, 368, 0.00005)
    train_model(10, 368, 0.00005)


     











  







