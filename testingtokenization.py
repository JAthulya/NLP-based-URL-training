
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

# ###set the random seed
# tf.random.set_seed(42)
# np.random.seed(42)



###read and process the data
df = pd.read_csv('finaldata.csv')
x= df['url']
y = df['label']

voc_size = 10000
messages = x.copy()

ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]',' ',urlparse(messages[i]).netloc)
    review = review.lower()
    review = review.split()
    review=' '.join(review)
    corpus.append(review)

onehot_repr = [nltk.word_tokenize(words) for words in corpus]
print(onehot_repr[0])
toknizer_en  = tf.keras.preprocessing.text.Tokenizer()
toknizer_en.fit_on_texts(onehot_repr)
sequences_en = toknizer_en.texts_to_sequences(onehot_repr)
print(sequences_en[0])

sent_length = 50
embedded_docs= pad_sequences(sequences_en,padding='pre',maxlen=sent_length)
print(embedded_docs[0])
embedded_docs = np.array(embedded_docs)

#x_final = np.array(embedded_docs)
x_final = embedded_docs
y_final  = np.array(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_final,y_final,test_size=0.20)


#make the model and train it
# embedding_vector_features=100
# model = Sequential()
# model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
# model.add(LSTM(100))
# model.add(Dense(1,activation='sigmoid'))
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model = Sequential()
model.add(LSTM(50,return_sequences=True, input_shape=(50,1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=64)

y_pred=model.predict(x_test) 
classes_y=np.round(y_pred).astype(int)
#model.save('E:\python\phishingurldetectionsystem\Models')
model.save("model4.h5")


# with open('model_pkl', 'wb') as files:
#     pickle.dump((model), files)
    
from sklearn.metrics import confusion_matrix
confusion_n = confusion_matrix(y_test,classes_y)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, classes_y))

# url = 'http://mail.google.com/a/camarabrejetuba.es.gov.br'
# test = "im checking this"
# # messages = url
# messages = urlparse(url).netloc
# print(messages)

# corpus=[]

# review = re.sub('[^a-zA-Z]',' ',test)
# review = review.lower()
# review = review.split()
# review=' '.join(review)
# corpus.append(review)
# print(corpus[0])

# onehot_repr = [nltk.word_tokenize(words) for words in corpus]
# #onehot_repr=[one_hot(words,voc_size)for words in corpus]
# print(onehot_repr[0])

# toknizer_en  = tf.keras.preprocessing.text.Tokenizer()
# toknizer_en.fit_on_texts(onehot_repr)
# sequences_en = toknizer_en.texts_to_sequences(onehot_repr)

# print(toknizer_en)
# print(sequences_en)