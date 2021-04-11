import random
import string

import pandas as pd
import numpy as np
import scipy
from scipy.stats import uniform

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

import wandb 


torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**28 - 1)
np.random.seed(hash("improves reproducibility") % 2**28 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**28 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**28 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Wandb Login
wandb.login()

config = dict(
    epochs = 10,
    batch_size = 80,
    learning_rate = 0.001,
    dataset = "Authorship 2000",
    architecture = "Dense:  Input, Layer 512, relu, batchnorm 512 , Layer 64, relu, batchnorm 64, dropout 0.1, output", 
    criterion = "BCEWithLogitsLoss",
    optimizer = "Adam"
)

class AuthorshipDataset(Dataset):

  def __init__(self, X_ngrams_data, X_punct_data, y_data):
    self.X_ngrams_data = X_ngrams_data
    self.X_punct_data = X_punct_data
    self.y_data = y_data

  
  def __getitem__(self, index):
    return self.X_ngrams_data[index], self.X_punct_data[index], self.y_data[index]


  def __len__(self):
    return len(self.y_data)

  def vector_size(self):
    return self.X_ngrams_data.shape[1], self.X_punct_data.shape[1]


class AuthorshipClassification(nn.Module):

  def __init__(self, input_size):
    super(AuthorshipClassification, self).__init__()

    print("input_size_ngrams:",input_size[0])
    self.layer1_ngrams = nn.Linear(input_size[0], 128)
    self.layer2_ngrams = nn.Linear(128, 64)
    self.layer3_ngrams = nn.Linear(64, 32)
    self.layer4_ngrams = nn.Linear(32, 16)
    self.layer5_ngrams = nn.Linear(16, 6)
 
    print("input_size_punct:",input_size[1])
    self.layer1_punct = nn.Linear(input_size[1], 16)
    self.layer2_punct = nn.Linear(16, 6)
    #self.layer3_punct = nn.Linear(4, 2)

    self.layer1_join = nn.Linear(12, 4)
    self.layer2_join = nn.Linear(4, 2)
    self.output_layer = nn.Linear(2, 1)

    self.selu = nn.SELU()
    self.dropout = nn.Dropout(p=0.1)
    self.batchnorm1 = nn.BatchNorm1d(128)
    #self.batchnorm2 = nn.BatchNorm1d(64)
    #self.batchnorm3 = nn.BatchNorm1d(32)
    #self.batchnorm4 = nn.BatchNorm1d(16)

  
  def forward(self, inputs_ngrams, inputs_punct):

    x_grams = self.layer1_ngrams(inputs_ngrams)
    x_grams = self.selu(x_grams)
    x_grams = self.batchnorm1(x_grams)
    #x_grams = self.dropout(x_grams)
    x_grams = self.layer2_ngrams(x_grams)
    x_grams = self.selu(x_grams)
    #x_grams = self.batchnorm2(x_grams)
    #x_grams = self.dropout(x_grams)
    x_grams = self.layer3_ngrams(x_grams)
    x_grams = self.selu(x_grams)
    #x_grams = self.batchnorm3(x_grams)
    #x_grams = self.dropout(x_grams)
    x_grams = self.layer4_ngrams(x_grams)
    x_grams = self.selu(x_grams)  
    #x_grams = self.batchnorm4(x_grams) 
    #x_grams = self.dropout(x_grams)  
    x_grams = self.layer5_ngrams(x_grams)
    x_grams = self.selu(x_grams)

    x_punct = self.layer1_punct(inputs_punct)
    x_punct = self.selu(x_punct)
    x_punct = self.layer2_punct(x_punct)
    x_punct = self.selu(x_punct)
    #x_punct = self.relu(x_punct)
    #x_punct = self.layer3_punct(x_punct)


    x = torch.cat((x_grams, x_punct), dim=1)

    x = self.layer1_join(x)
    x = self.selu(x)
    x = self.layer2_join(x)
    x = self.selu(x)
    output = self.output_layer(x)

    return output


def model_pipeline(hyperparameters):

  with wandb.init(project="authorship", config=hyperparameters):

    config = wandb.config
    print("Calling make")
    model, train_loader, test_loader, criterion, optimizer = make(config)
    print(model)

    print("Calling train")
    train(model, train_loader, criterion, optimizer, config)

    print("Calling test")
    return test(model, test_loader)



def get_data(type_data="train"):
  TEMP_DATA_DIR = "data/small/"

  if type_data == "train":
    X_train_ngrams  = np.memmap(TEMP_DATA_DIR + 'features_ngrams_X_train.npy', dtype='float32', mode='r', shape=(35768, 44872))
    X_train_punct = np.memmap(TEMP_DATA_DIR + 'features_punct_X_train.npy', dtype='float32', mode='r', shape=(35768, 32))
    Y_train = np.memmap(TEMP_DATA_DIR + 'Y_train.npy', dtype='int32', mode='r', shape=(35768))
    return AuthorshipDataset(torch.from_numpy(X_train_ngrams), 
                             torch.from_numpy(X_train_punct),
                             torch.from_numpy(Y_train.astype('float32')))
  elif type_data == "test":
    X_test_ngrams = np.memmap(TEMP_DATA_DIR + 'features_ngrams_X_dev.npy', dtype='float32', mode='r', shape=(8942, 44872))
    X_test_punct = np.memmap(TEMP_DATA_DIR + 'features_punct_X_dev.npy', dtype='float32', mode='r', shape=(8942, 32))
    Y_test = np.memmap(TEMP_DATA_DIR + 'Y_dev.npy', dtype='int32', mode='r', shape=(8942))
    return AuthorshipDataset(torch.from_numpy(X_test_ngrams), 
                             torch.from_numpy(X_test_punct),
                             torch.from_numpy(Y_test.astype('float32')))



def make(config):

  # get_data
  data_train = get_data(type_data="train")
  data_test = get_data(type_data="test")

  data_input_size = data_train.vector_size()
  
  # data_loaders
  train_loader = DataLoader(dataset=data_train, batch_size=config.batch_size, shuffle=False)
  test_loader = DataLoader(dataset=data_test, batch_size=config.batch_size, shuffle=False)

  
  #model
  model = AuthorshipClassification(data_input_size).to(device)

  # criterion and optimizer
  criterion = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)


  return model, train_loader, test_loader, criterion, optimizer


def binary_accuracy(y_pred, y_test):

  y_pred_tag = torch.round(torch.sigmoid(y_pred))
  correct_results = (y_pred_tag == y_test).sum().float()
  acc = correct_results / y_test.shape[0]
  return acc


def train(model, train_loader, criterion, optimizer, config):

  # Tell wandb to watch 
  wandb.watch(model, criterion, log_freq=10)

  model.train()
  for epoch in range(1, config.epochs+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_ngrams_batch, X_punct_batch, y_batch in train_loader:
      X_ngrams_batch, X_punct_batch, y_batch = (X_ngrams_batch.to(device), 
                                                X_punct_batch.to(device), 
                                                y_batch.to(device))
      optimizer.zero_grad()

      y_pred = model(X_ngrams_batch, X_punct_batch)
  

      loss = criterion(y_pred, y_batch.unsqueeze(1))
      acc = binary_accuracy(y_pred, y_batch.unsqueeze(1))

      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()
      epoch_acc += acc.item()
    print(f'Epoch {epoch}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')  
    wandb.log({
          "Epoch": epoch,
          "Train Accuracy": epoch_acc/len(train_loader),
          "Train Loss": epoch_loss/len(train_loader)})


def test(model, test_loader):
    model.eval()
    y_pred_list = []
    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        for X_ngrams_batch, X_punct_batch, y_batch in test_loader:
            X_ngrams_batch,X_punct_batch ,y_batch = (X_ngrams_batch.to(device), 
                                                    X_punct_batch.to(device), 
                                                    y_batch.to(device))
            outputs = model(X_ngrams_batch, X_punct_batch)
            y_test_pred = torch.sigmoid(outputs)
           
            predicted = torch.round(y_test_pred).squeeze()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            print(f"Accuracy of the model on the {total} " +
              f"test data: {100 * correct / total}%")


            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.extend(y_pred_tag.cpu().numpy())



            
        wandb.log({"test_accuracy": correct / total})

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    torch.onnx.export(model,(X_ngrams_batch, X_punct_batch),"model.onnx")
    #wandb.save("model.onnx")
    return y_pred_list