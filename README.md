# Authorship Verification with neural networks via stylometric feature concatenation

This software was developed for the PAN 2021 author verification task. 

Our work focuses on extracting two stylometric features, character-level n-grams and the use of punctuation marks in the texts. Subsequently, we train a neural network with each of them and finally combine them into a final neural network for the classifier decision making. It is described in :

Antonio Menta and Ana Garcia Serrano.  Authorship Verification with neural networks via stylometric feature concatenation, Notebook for PAN at CLEF 2021. In Proceedings of the Working Notes of CLEF 2021 - Conference and Labs of the Evaluation Forum. Bucharest, Romania, September 2021.CEUR-WS.org.
 https://pan.webis.de/downloads/publications/papers/menta_2021.pdf

 

Steps to replicate the results: 

1. Clone this repository
2. Install dependencies:
<pre><code>pip install -r requirements.txt</code></pre>
3. For training purpose:
<pre><code>python train_small_dataset.py</code></pre>
4. For test purpose:
<pre><code>python predict_small_dataset.py</code></pre>

Same for larger version. Be carefull with memory issues. 



# Data

Download the dataset from https://pan.webis.de/data.html to data folder.

# Citing
@InProceedings{menta:2021,
  author =              {Menta, Antonio and Garcia-Serrano, Ana},
  booktitle =           {{CLEF 2021 Labs and Workshops, Notebook Papers}},
  crossref =            {pan:2021},
  editor =              {},
  month =               sep,
  publisher =           {CEUR-WS.org},
  title =               {{Authorship verification with neural networks via stylometric feature concatenation}},
  url =                 {},
  year =                2021
}

