# -*- coding: utf-8 -*-
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, Model
from keras.optimizers import Adam

import numpy as np
import pandas as pd
import sklearn
import gensim
from gensim.models import KeyedVectors

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb
from keras.utils import np_utils
import time
import pickle
import matplotlib.pyplot as plt
import h5py


t1=time.time()
df_train_txt = pd.read_csv('training_text', sep='\|\|', header=None, skiprows=1, names=["ID","Text"], engine='python')
df_train_var = pd.read_csv('training_variants')

test1x = pd.read_csv('test_text', sep='\|\|', header=None, skiprows=1, engine='python',names=["ID","Text"])
test1 = pd.read_csv('test_variants')

filtered = pd.read_csv('stage1_solution_filtered.csv')
test2 = pd.read_csv('stage2_test_variants.csv')
test2x = pd.read_csv('stage2_test_text.csv', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
def whichClass(lst):
    for i, n in enumerate(lst):
        if n==1:
            return i+1
filtered['Class'] = filtered.loc[:,'class1':'class9'].apply(whichClass, axis=1)
test1 = pd.merge(test1, filtered.loc[:,['ID', 'Class']], on='ID').fillna('')
test1 = pd.merge(test1, test1x, on='ID').fillna('')
df_train = pd.merge(df_train_var, df_train_txt, how='left', on='ID').fillna('')
df_train = df_train.append(test1)
df_train.reset_index(drop=True, inplace=True)
# df_train = df_train.iloc[:300,:]

df_test = pd.merge(test2, test2x, how='left', on='ID').fillna('')
# df_test = df_test.iloc[:10,:]

w2v_model = KeyedVectors.load_word2vec_format('PubMed-and-PMC-w2v.bin', binary=True)

# word2vec embeding
word_vec = []
train_text = list(df_train['Text'])
for i in train_text:
    case_vec = []
    for j in i.split(' '):
        if j in w2v_model:
            case_vec.append(w2v_model[j])
        else:
            pass
    word_vec.append(case_vec)
word_vec = np.array(word_vec)
word_vec_pad = []
for i in word_vec:
    if len(i)>10000:
        word_vec_pad.append(i[:10000])
    if len(i)<10000:
        for j in range(10000-len(i)):
            i.append(np.array([0 for k in range(200)]))
        word_vec_pad.append(i)

word_vec_pad = np.array(word_vec_pad)

test_text = list(df_test['Text'])
word_vec = []
for i in test_text:
    case_vec = []
    for j in i.split(' '):
        if j in w2v_model:
            case_vec.append(w2v_model[j])
        else:
            pass
    word_vec.append(case_vec)

word_vec_pad_test = []
for i in word_vec:
    if len(i)>10000:
        word_vec_pad_test.append(i[:10000])
    if len(i)<10000:
        for j in range(10000-len(i)):
            i.append(np.array([0 for k in range(200)]))
        word_vec_pad_test.append(i)

# encode class values as integers
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(list(df_train['Class']))
encoded_Y = encoder.transform(list(df_train['Class']))
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

model = Sequential()
model.add(Bidirectional(LSTM(64), input_shape=(10000, 200), name='bi-lstm'))
model.add(Dropout(0.5))
model.add(Dense(9, activation='softmax'))
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy','categorical_crossentropy'])
history = model.fit(word_vec_pad, dummy_y, batch_size=64, epochs=10, validation_split=0.2)

pickle.dump(history.history['loss'], open('history_loss_train', 'wb'))
pickle.dump(history.history['val_loss'], open('history_loss_val', 'wb'))
pickle.dump(history.history, open('history_all', 'wb'))


word_vec_pad_test = np.array(word_vec_pad_test)
model_output = model.predict(word_vec_pad_test)
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('bi-lstm').output)
intermediate_output = intermediate_layer_model.predict(word_vec_pad_test)
pickle.dump(model_output, open('model_output', 'wb'))
pickle.dump(intermediate_output, open('inter_output', 'wb'))

model.save('lstm_model.h5')

t2 =time.time()
print (t2-t1)/60.0, t2-t1
