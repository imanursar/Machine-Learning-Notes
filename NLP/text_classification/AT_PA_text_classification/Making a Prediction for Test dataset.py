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

# function
import sys
sys.path.append('function/')
from ursar import nlp


# In[2]:


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

def predict_embedding(reviews, model,input):
    # apply preprocess_text function to out training dataset
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

def print_score (y_test,y_pred,y_probs):
    print("comfusion matrix = ")
    print(confusion_matrix(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    (tn,fp,fn,tp) = confusion_matrix(y_test, y_pred).ravel()

    print("")
    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy_score = ', accuracy)
    bas = balanced_accuracy_score(y_test, y_pred)
    print('balanced_accuracy_score = ', bas)
    #balanced accuracy is equal to the arithmetic mean of sensitivity (true positive rate) and specificity (true negative rate),
    #or the area under the ROC curve with binary predictions rather than scores.

    #In multilabel classification,
    #this function computes subset accuracy: the set of labels predicted for
    #a sample must exactly match the corresponding set of labels in y_true

    print("")
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    aps = average_precision_score(y_test, y_pred)
    print ("precision score = ", precision)
    print ("average precision score = ", aps)
    print ("recall score = ", recall)

    #precision An interesting one to look at is the accuracy of the positive pre‐ dictions; this is called the precision of the classifier
    # recall, also called sensitivity or true positive rate (TPR): this is the ratio of positive instances that are correctly detected by the classifier
    #precision = TP/TP + FP
    #recall = TP/TP + FN

    print("")
    f1 = f1_score(y_test, y_pred)
    print ("F1 score = ", f1)
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    aucs = auc(recall, precision)
    print ("AUC of Precision-Recall Curve on Testing = ", aucs)
    aucroc = roc_auc_score(y_test,y_probs)
    print ("AUC of ROC = ", aucroc)
    gini = aucs*2 - 1
    print("Gini = ", gini)

    print("")
    cr = classification_report(y_test,y_pred)
    print("classification_report")
    print(cr)

    #The F1 score is the harmonic mean of precision and recall (Equation 3-3).
    #Whereas the regular mean treats all values equally,
    #the harmonic mean gives much more weight to low values.


# # Load Test dataset

# In[3]:


print("Load Test dataset")
test = pd.read_csv('DATA/test_data_restaurant.tsv', sep='\t',header=None).sample(frac=1).reset_index(drop=True)
test.columns = ['sentence', 'label']
print("\ndata testing shape")
print(test.shape)


# In[4]:


print("\nis the test dataset contain the null values?")
print(test.isnull().sum())


# In[5]:


test["label"] = list(map(lambda x: 1 if x=="positive" else 0, test["label"]))
print("\nchange label for test dataset")
test["label"].unique()
y_test = test["label"]


# In[6]:


print("\napply preprocess_text function to out testing dataset")
reviews_test = []
sentences = list(test["sentence"])
for sen in sentences:
    reviews_test.append(preprocess_text(sen))


# In[9]:


print("\nmake token for test dataset")
token_test_ann = tokenize_matrix(reviews_test,"freq")
token_test = tokenize_embedding(reviews_test,'post', 120)


# In[10]:


print("\nload matrix_ANN model")
model = load_model('model/model_matrix_ANN.h5')


# In[11]:


loss, accuracy = model.evaluate(token_test_ann,y_test , verbose=False)
print("\nTesting Accuracy:  {:.4f}".format(accuracy))
print("Testing loss:  {:.4f}".format(loss))


# In[12]:


print("\nload CNN model")
model = load_model('model/model_CNN.h5')


# In[13]:


loss, accuracy = model.evaluate(token_test,y_test, verbose=False)
print("\nTesting Accuracy:  {:.4f}".format(accuracy))
print("Testing loss:  {:.4f}".format(loss))


# In[14]:


print("\nload CNN LSTM model")
model = load_model('model/model_CNN_LSTM.h5')


# In[15]:


loss, accuracy = model.evaluate(token_test,y_test, verbose=False)
print("\nTesting Accuracy:  {:.4f}".format(accuracy))
print("Testing loss:  {:.4f}".format(loss))


# In[16]:


print("\nload CNN wiki model")
model = load_model('model/model_CNN_wiki.h5')


# In[17]:


loss, accuracy = model.evaluate(token_test,y_test, verbose=False)
print("\nTesting Accuracy:  {:.4f}".format(accuracy))
print("Testing loss:  {:.4f}".format(loss))


# In[18]:


print("\nload CNN LSTM wiki model")
model = load_model('model/model_CNN_LSTM_wiki.h5')


# In[19]:


loss, accuracy = model.evaluate(token_test,y_test, verbose=False)
print("\nTesting Accuracy:  {:.4f}".format(accuracy))
print("Testing loss:  {:.4f}".format(loss))


# In[ ]:





# In[ ]:




