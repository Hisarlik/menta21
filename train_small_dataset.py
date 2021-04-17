import os
import pandas as pd
from model import model_pipeline, predict_model
from vectorizer import vectorize_dataset
from sklearn.model_selection import train_test_split



#path_training_small_truth = "data/pan20-authorship-verification-training-small-truth.jsonl"
#path_training_small = "data/pan20-authorship-verification-training-small.jsonl"
#path_training_large_truth = "data/pan20-authorship-verification-training-large-truth.jsonl"
#path_training_large = "data/pan20-authorship-verification-training-large.jsonl"


sklearn_random = 20

def store_pandas(dataframe, config):


    path = config['path_dataset']+"train.csv"

    # if file does not exist write header 
    if not os.path.isfile(path):
       dataframe.to_csv(path, index=False)
    else: # else it exists so append without writing the header
       dataframe.to_csv(path, mode='a', header=False, index=False)


def create_dataset(config):
    df_texts = pd.read_json(config['path_training'], lines=True, chunksize=1000)
    df_truth = pd.read_json(config['path_training_truth'], lines=True, chunksize=1000)

    i =0
    for texts, truth in zip(df_texts, df_truth):
        print(len(texts))
        df_join_training_data = pd.concat([truth, texts], axis=1).reindex(truth.index)
        df_join_training_data = df_join_training_data.loc[:,~df_join_training_data.columns.duplicated()]
        df_join_training_data[['text1','text2']] = pd.DataFrame(df_join_training_data.pair.tolist(), index= df_join_training_data.index)
        df_join_training_data = df_join_training_data.drop(columns=["pair", "fandoms","authors"])

        print(f"save chunk [{i*1000},{(i+1)*1000}]")
        i += 1
        store_pandas(df_join_training_data, config)
    
    df_texts=None
    df_truth=None

    randomize_dataset(config)


def randomize_dataset(config):

    path = config['path_dataset']+"train.csv"

    dataset = pd.read_csv(path)

    dataset = dataset.sample(frac=1).reset_index(drop=True)

    limit_dataset = config.get("limit_dataset")
    if limit_dataset:
        dataset = dataset[:limit_dataset]




    print(dataset.head())
    print("save dataset")
    dataset.to_csv(config['path_dataset']+"train.csv", index=False)

    dataset=None





if __name__ == "__main__":

    

    config = dict(
        path_training="data/pan20-authorship-verification-training-small.jsonl",
        path_training_truth="data/pan20-authorship-verification-training-small-truth.jsonl",
        epochs = 10,
        batch_size = 128,
        learning_rate = 0.001,
        dataset = "Authorship 2000",
        architecture = "Dense:  Input, Layer 512, relu, batchnorm 512 , Layer 64, relu, batchnorm 64, dropout 0.1, output", 
        criterion = "BCEWithLogitsLoss",
        optimizer = "Adam",
        limit_dataset = None,
        path_dataset = "data/temp/",
        limit_data_vectorizer = 15000
        
    )

    task = "train"

    if task == "train":

        create_dataset(config)
        vectorize_dataset(config)
        model_pipeline(config)

    elif task == "predict":

        predict_model(config)




