import pandas as pd
import pickle

df = pd.read_csv('finaldata.csv')
print(df.head())
x= df['url']
y = df['label']
#print(x.shape)
#print(y.shape)

import tensorflow as tf

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

voc_size = 10000
messages = x.copy()
#print(messages[1])
#messages.reset_index(inplace=True)

import nltk
import re
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages[i])
    review = review.lower()
    review = review.split()
    review=' '.join(review)
    corpus.append(review)
#print(corpus[11])
onehot_repr=[one_hot(words,voc_size)for words in corpus]
#print(onehot_repr[1])

sent_length = 50
embedded_docs= pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs[55639])
print(corpus[55639])

model = Sequential()
model.add(LSTM(50,return_sequences=True, input_shape=(50,1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

#print(len(embedded_docs))
import numpy as np
x_final = np.array(embedded_docs)
y_final  = np.array(y)

print(x_final.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_final,y_final,test_size=0.20,random_state=42)


model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=64)

#y_pred = model.predict_classes(x_test)
#y_pred= model.predict_classes(x_test)

y_pred=model.predict(x_test) 
classes_y=np.round(y_pred).astype(int)

model.save('model1.h5')
#pickle.dump(model, open('model6.pkl','wb'))


# with open("model5.pkl", "wb") as file:
#      pickle.dump(model, file)


    
from sklearn.metrics import confusion_matrix
confusion_n = confusion_matrix(y_test,classes_y)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, classes_y))
