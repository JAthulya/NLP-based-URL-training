
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import pandas as pd
import pickle
from urllib.parse import urlparse, urlencode

import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

from numpy.random import seed 
seed(42)
from tensorflow.random import set_seed
set_seed(42)

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np

model = tf.keras.models.load_model("model4.h5")

url = 'https://www.vakantieverhuur.be/js/scriptaculous/qhc/office365/3dd9fc82aa3b4b755b1f1457f4bb3b04/pass.php'
#messages = url
messages = urlparse(url).netloc
print(messages)

corpus=[]

review = re.sub('[^a-zA-Z]',' ',messages)
review = review.lower()
review = review.split()
review=' '.join(review)
corpus.append(review)
print(corpus[0])

onehot_repr = [nltk.word_tokenize(words) for words in corpus]
#onehot_repr=[one_hot(words,voc_size)for words in corpus]
print(onehot_repr[0])

toknizer_en  = tf.keras.preprocessing.text.Tokenizer()
toknizer_en.fit_on_texts(onehot_repr)
sequences_en = toknizer_en.texts_to_sequences(onehot_repr)

print(toknizer_en)
print(sequences_en)

sent_length = 50
embedded_docs= pad_sequences(sequences_en,padding='pre',maxlen=sent_length)
print(embedded_docs[0])

embedded_docs = np.array(embedded_docs)
print(embedded_docs)

x_test = embedded_docs
#y_final  = np.array(y)
print(x_test)

y_pred = model.predict(x_test)
classes_y=np.round(y_pred).astype(int)

print(y_pred)
print(classes_y)