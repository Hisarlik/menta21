import pandas as pd
from model import model_pipeline
from vectorizer import vectorize_dataset
from sklearn.model_selection import train_test_split



path_training_small_truth = "data/pan20-authorship-verification-training-small-truth.jsonl"
path_training_small = "data/pan20-authorship-verification-training-small.jsonl"


sklearn_random = 20



def create_dataset(config):
    df_texts = pd.read_json(path_training_small, lines=True)
    df_truth = pd.read_json(path_training_small_truth, lines=True)

    df_join_training_data = pd.concat([df_truth, df_texts], axis=1).reindex(df_truth.index)
    df_join_training_data = df_join_training_data.loc[:,~df_join_training_data.columns.duplicated()]

    df_join_training_data[['text1','text2']] = pd.DataFrame(df_join_training_data.pair.tolist(), index= df_join_training_data.index)
    df_join_training_data[['author1','author2']] = pd.DataFrame(df_join_training_data.authors.tolist(), index= df_join_training_data.index)

    df_join_training_data = df_join_training_data.drop(columns=["pair", "fandoms","authors"])

    dataset = df_join_training_data.sample(frac=1).reset_index(drop=True)

    limit_dataset = config.get("limit_dataset")
    if limit_dataset:
        dataset = dataset[:limit_dataset]




    print(dataset.head())

    train, test = train_test_split(dataset, test_size=0.15, random_state=sklearn_random)

    train, dev = train_test_split(train, test_size=0.20, random_state=sklearn_random)

    train.to_csv(config['path_dataset']+"train.csv", index=False)
    dev.to_csv(config['path_dataset']+"dev.csv", index=False)
    test.to_csv(config['path_dataset']+"test.csv", index=False)






if __name__ == "__main__":

    

    config = dict(
        epochs = 10,
        batch_size = 80,
        learning_rate = 0.001,
        dataset = "Authorship 2000",
        architecture = "Dense:  Input, Layer 512, relu, batchnorm 512 , Layer 64, relu, batchnorm 64, dropout 0.1, output", 
        criterion = "BCEWithLogitsLoss",
        optimizer = "Adam",
        limit_dataset = 500,
        path_dataset = "data/500/"
        
    )


    #create_dataset(config)
    #vectorize_dataset(config)
    model_pipeline(config)




