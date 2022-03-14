import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['PYTHONHASHSEED'] = '0'

import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
import urllib
from urllib.parse import urlparse, urlencode
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from joblib import dump, load

tf.random.set_seed(42)
np.random.seed(42)

#model = tf.keras.models.load_model('E:\python\phishingurldetectionsystem\Models')
model = tf.keras.models.load_model("model1.h5")

import nltk
import re
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()  

voc_size = 10000
url = 'https://support.office.com/ar-sa/article/%D8%A7%D9%84%D8%AF%D8%A7%D9%84%D8%A9-FORECAST-ETS-SEASONALITY-32A27A3B-D22F-42CE-8C5D-EF3649269F3C'
# messages = url
messages = urlparse(url).netloc
print(messages)

corpus=[]

review = re.sub('[^a-zA-Z]',' ',messages)
review = review.lower()
review = review.split()
review=' '.join(review)
corpus.append(review)
print(corpus[0])

with open('mapping.pkl', 'rb') as fout:
  mapping = pickle.load(fout)
  
clf = load('map.joblib')

# mapping = pickle.load(fout)

# onehot_repr = [nltk.word_tokenize(words) for words in corpus]
#corpus, onehot_repr =  zip(*mapping)
#print(mapping)

onehot_repr=[mapping(words,voc_size)for words in corpus]

#mapping = {c:o for c,o in zip(corpus, onehot_repr)}
print(onehot_repr[0])

sent_length = 50
embedded_docs= pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
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
'''
if classes_y == 0:
    print('not a phishing')
else:
    print('phishing')
    '''
