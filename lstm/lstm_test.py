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
# load test data
test2 = pd.read_csv('stage2_test_variants.csv')
test2x = pd.read_csv('stage2_test_text.csv', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
df_test = pd.merge(test2, test2x, how='left', on='ID').fillna('')
w2v_model = KeyedVectors.load_word2vec_format('PubMed-and-PMC-w2v.bin', binary=True)
test_text = list(df_test['Text'])

# embedding
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

# load model
filename = 'lstm_model.h5'
model = load_model(filename)

# test
word_vec_pad_test = np.array(word_vec_pad_test)
model_output = model.predict(word_vec_pad_test)
pickle.dump(model_output, open('model_output_allTest', 'wb'))

intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('bi-lstm').output)
intermediate_output = intermediate_layer_model.predict(word_vec_pad_test)
pickle.dump(intermediate_output, open('inter_output_allTest', 'wb'))


t2 =time.time()
print (t2-t1)/60.0, t2-t1