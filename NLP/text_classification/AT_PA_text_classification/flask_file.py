#Library
from flask import Flask, request, jsonify, json, make_response
from flask_cors import CORS
import traceback


import pandas as pd
import numpy as np
import pickle
import datetime

import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
from tensorflow.keras.models import load_model
# NLP library
import nltk
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# function
import sys
sys.path.append('function/')
from ursar import nlp

# API definition
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] =False
CORS(app)

#function
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

def tokenize_matrix(corpus,mode):
    #corpus, 5000, 'post', 120
    # create the tokenizer
    # load label train dataset file here
    with open('model/tokenizer_ann', 'rb') as picklefile:
        tokenizer = pickle.load(picklefile)

    # encode training data set
    sen = tokenizer.texts_to_matrix(corpus, mode=mode)
    return(sen)

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

def predict_sentiment(review, model,input):
    # apply preprocess_text function to out training dataset
    reviews = []
    reviews.append(preprocess_text(review))
    # encode
    if (input == "embedding"):
        encoded = tokenize_embedding(reviews,'post', 120)
    if (input == "matrixs"):
        encoded = tokenize_matrix(reviews,"freq")
    # prediction
    yhat = model.predict(encoded, verbose=0)

    if (yhat[0,0]>=0.5):
        res = "positive review"
    else:
        res = "negative review"
    return (res,yhat[0,0])

#load pickle, model and others
ann_model = load_model('model/model_matrix_ANN.h5')
cnn_model = load_model('model/model_CNN.h5')
cnn_lstm_model = load_model('model/model_CNN_LSTM.h5')
cnn_wiki_model = load_model('model/model_CNN_wiki.h5')
cnn_lstm_wiki_model = load_model('model/model_CNN_LSTM_wiki.h5')

#routing to URL
@app.route('/sentiment_analysis', methods=['POST'])

#main program
def template_flash():
    try:
        json_ = request.json

        sent = json_["sentence"]

        result = dict()
        result['sentence'] = sent

        res, prob = predict_sentiment(sent,ann_model,"matrixs")
        result['result from ann_model'] = res
        result['probability from ann_model'] = round(prob.tolist(), 4)

        res, prob = predict_sentiment(sent,cnn_model,"embedding")
        result['result from cnn_model'] = res
        result['probability from cnn_model'] = round(prob.tolist(), 4)

        res, prob = predict_sentiment(sent,cnn_lstm_model,"embedding")
        result['result from cnn_lstm_model'] = res
        result['probability from cnn_lstm_model'] = round(prob.tolist(), 4)

        res, prob = predict_sentiment(sent,cnn_wiki_model,"embedding")
        result['result from cnn model wiki corpus'] = res
        result['probability from cnn model wiki corpus'] = round(prob.tolist(), 4)

        res, prob = predict_sentiment(sent,cnn_lstm_wiki_model,"embedding")
        result['result from cnn lstm model wiki corpus'] = res
        result['probability from cnn lstm model wiki corpus'] = round(prob.tolist(), 4)

        return jsonify(result)

    except:
        return jsonify({'trace': traceback.format_exc()})

if __name__ == '__main__':
    port = 2021
    app.run(host='localhost', port=port, debug=True)
