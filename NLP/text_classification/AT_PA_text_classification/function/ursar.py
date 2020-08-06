# common library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import pickle
import functools
import operator

# NLP library
import nltk
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class nlp:
    def __init__(self, main_data):
        print("ursar")

    def preprocess_text(sentence):
        # load stopwords for bahasa in NLTK
        # nltk.download('stopwords')
        id_stop = set(nltk.corpus.stopwords.words('indonesian'))

		# load stemming modul from Sastrawi
        factory_Stemmer = StemmerFactory()
        stemmer = factory_Stemmer.create_stemmer()

		# Remove all the special characters
		# remove all the non-word characters (letters and numbers) from a string and keep the remaining characters
        sentence = re.sub(r'\W', ' ', str(sentence))

		# Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)

		# Single character removal
		# Sometimes removing punctuation marks, such as an apostrophe, results in a single character which has no meaning
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

		# Remove single characters from the start
        sentence = re.sub(r'\^[a-zA-Z]\s+', ' ', sentence)

		# Remove enter type to space
        sentence = sentence.replace("\n"," ")

		# Substituting multiple spaces with single space
        sentence = re.sub(r'\s+', ' ', sentence)

		#removes spaces from at the end
        sentence = re.sub(r"\s+$", "", sentence)

		# Converting to Lowercase
        sentence = sentence.lower()

		# split our list for NLTK
        sentence = sentence.split()

		# stop word from NLTK
        sentence = [word for word in sentence if word not in id_stop]

		# words with length less than 4, have also been removed.
        sentence = [word for word in sentence if len(word) > 3]

		# combine our sentence for Sastrawi
        sentence = ' '.join(sentence)

		# Steeeming = reduce the word into dictionary root form
		# Stemming with Python Sastrawi
        sentence = stemmer.stem(sentence)
        return sentence

    def plot_history(history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

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

        # print("")
        # ear = ((fp+tp)/(fp+tp+tn+fn)) #mirip akurasi tetapi fp+tp
        # print ("Expected Approval Rate = ", ear)
        # edr = (fp/(fp+tp)) #contamination
        # print ("Expected Default Rate = ", edr)

        print("")
        cr = classification_report(y_test,y_pred)
        print("classification_report")
        print(cr)

        #The F1 score is the harmonic mean of precision and recall (Equation 3-3).
        #Whereas the regular mean treats all values equally,
        #the harmonic mean gives much more weight to low values.
