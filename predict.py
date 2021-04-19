import os
import argparse
import pandas as pd
from model import model_pipeline, predict_model
from vectorizer import vectorize_dataset, vectorize_predict
from sklearn.model_selection import train_test_split



sklearn_random = 20

def store_pandas(dataframe, path):


    # if file does not exist write header 
    if not os.path.isfile(path):
       dataframe.to_csv(path, index=False)
    else: # else it exists so append without writing the header
       dataframe.to_csv(path, mode='a', header=False, index=False)


def delete_file(filePath):

    # As file at filePath is deleted now, so we should check if file exists or not not before deleting them
    if os.path.exists(filePath):
        os.remove(filePath)
    else:
        print("Can not delete the file as it doesn't exists")

def parse_predict_data(config):
    df_texts = pd.read_json(config['path_data'], lines=True, chunksize=50)    



    path_predict = config['path_predict']+"predict.csv"


    path_temp = config['path_model']+"temp.csv"
    delete_file(path_temp)
    delete_file(path_predict)
    for chunk in df_texts:
        chunk[['text1','text2']] = pd.DataFrame(chunk.pair.tolist(), index= chunk.index)
        chunk = chunk.drop(columns=["pair", "fandoms"])
        store_pandas(chunk, path_temp)
        store_pandas(chunk["id"], path_predict)       





if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Predict script')
    parser.add_argument('-i', type=str,
                        help='Path to the jsonl-file with data')
    parser.add_argument('-o', type=str, 
                        help='Path to output files')
    args = parser.parse_args()

    # validate:
    if not args.i:
        raise ValueError('The data path is required')
    if not args.o:
        raise ValueError('The output folder path is required')
    
    # load:
    pairs = f"{args.i}/pairs.jsonl"
    pred = f"{args.o}/"

    

    config = dict(
        path_data= pairs,
        path_model = "data/500/",
        path_predict = pred,
        batch_size = 128

        
    )

    parse_predict_data(config)
    vectorize_predict(config)
    predict_model(config)




