# Authorship Verification with neural networks via stylometric feature concatenation

This software was developed for the PAN 2021 author verification task. 

Our work focuses on extracting two stylometric features, character-level n-grams and the use of punctuation marks in the texts. Subsequently, we train a neural network with each of them and finally combine them into a final neural network for the classifier decision making. It is described in :

Antonio Menta and Ana Garcia Serrano.  Authorship Verification with neural networks via stylometric feature concatenation, Notebook for PAN at CLEF 2021. In Proceedings of the Working Notes of CLEF 2021 - Conference and Labs of the Evaluation Forum. Bucharest, Romania, September 2021.
 http://ceur-ws.org/Vol-2936/paper-181.pdf

 

Steps to replicate the results: 

1. Clone this repository
2. Install dependencies:
    pip install -r requirements.txt
3. For training purpose:
    python train_small_dataset.py 
4. For test purpose:
    python predict_small_dataset.py


# Data

Download the dataset from https://pan.webis.de/data.html to data folder.


