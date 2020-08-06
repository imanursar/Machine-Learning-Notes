#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
from tensorflow.keras.models import load_model
# NLP library
import nltk
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[9]:


def preprocess_text(sentence):
    id_stop = set(nltk.corpus.stopwords.words('indonesian'))
    factory_Stemmer = StemmerFactory()
    stemmer = factory_Stemmer.create_stemmer()
    sentence = re.sub(r'\W', ' ', str(sentence))
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\^[a-zA-Z]\s+', ' ', sentence)
    sentence = sentence.replace("\n"," ")
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub(r"\s+$", "", sentence)
    sentence = re.sub(r"^\s+", "", sentence)
    sentence = sentence.lower()
    sentence = sentence.split()
    sentence = [word for word in sentence if word not in id_stop]
    sentence = [word for word in sentence if len(word) > 3]
    sentence = ' '.join(sentence)
    sentence = stemmer.stem(sentence)
    return sentence

def tokenize_embedding(corpus,padding_type,max_length):
    #corpus, 5000, 'post', 120
    # create the tokenizer
    # load label train dataset file here
    with open('model/tokenizer_embed', 'rb') as picklefile:
        tokenizer = pickle.load(picklefile)
    # encode training data set
    sen = tokenizer.texts_to_sequences(corpus)
    sen = pad_sequences(sen, padding=padding_type, maxlen=max_length)
    return(sen)
        
def tokenize_matrix(corpus,mode):
    #corpus, 5000, 'post', 120
    # create the tokenizer
    # load label train dataset file here
    with open('model/tokenizer_ann', 'rb') as picklefile:
        tokenizer = pickle.load(picklefile)
    # encode training data set
    sen = tokenizer.texts_to_matrix(corpus, mode=mode)
    return(sen)


# In[3]:


# classify a review as negative (0) or positive (1)
def predict_sentiment(review, model):
    # apply preprocess_text function to out training dataset
    reviews = []
    print(review)
    reviews.append(nlp.preprocess_text(review))
    # encode
    encoded = nlp.tokenize_embedding(reviews,'post', 120)
    # prediction
    yhat = model.predict(encoded, verbose=0)
    if (yhat[0,0]>=0.5):
        res = "positive review"
    else:
        res = "negative review"
    return (res,yhat[0,0])


# In[4]:


# load model
cnn_lstm_model = load_model('model/model_CNN_LSTM.h5')


# In[10]:


# test positive text
text = 'Saya telah mencoba pasta dan pizza yang mereka hidangkan, rasanya…. Enak!'
print(predict_sentiment(text, cnn_lstm_model,"embedding"))


# In[ ]:




