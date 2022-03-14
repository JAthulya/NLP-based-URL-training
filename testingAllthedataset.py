import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import tensorflow as tf
import numpy as np
from urllib.parse import urlparse, urlencode
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
tf.random.set_seed(42)
np.random.seed(42)

model = tf.keras.models.load_model('model1.h5')

#df = pd.read_csv('5.urldata - Copy (2).csv')
df = pd.read_csv('finaldata.csv')
#print(df.head())
x = df['url']
y = df['label']
#print(x.shape)
#print(y.shape)

voc_size = 10000
messages = x.copy()

import nltk
import re
from nltk.corpus import stopwords



from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]',' ',urlparse(messages[i]).netloc)
    review = review.lower()
    review = review.split()
    review=' '.join(review)
    corpus.append(review)
    
#print(corpus[0])
onehot_repr=[one_hot(words,voc_size)for words in corpus]
#print(onehot_repr[1])
sent_length = 50
embedded_docs= pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
#print(embedded_docs[1])


x_final = np.array(embedded_docs)
y_final  = np.array(y)

y_pred = model.predict(x_final)
classes_y=np.round(y_pred).astype(int)

from sklearn.metrics import confusion_matrix
confusion_n = confusion_matrix(y,classes_y)
from sklearn.metrics import accuracy_score
print(accuracy_score(y, classes_y))















