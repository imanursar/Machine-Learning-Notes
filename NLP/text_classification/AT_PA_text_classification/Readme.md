# Sentiment analysis: restaurant reviews

Problem Definition
==================

Understanding customer voices is very important for a company to improve the quality of products and services. One of the ways is by identifying the sentiments of customer voices. Thus, the company could know whether its products or services are viewed positively or negatively.

Motivated by the above background, Prosa.ai will create an API service for sentiment analysis. Thus, Prosa.ai must develop an AI model that predicts the sentiment of a text whether it is positive or negative. Because deep learning methods achieved better performance in general, Prosa.ai decided to develop the AI model by using one of deep learning methods.

Solution
========

To solve this problem, we will follow the typical machine learning pipeline. We will first import the required libraries and the dataset. We will then do exploratory data analysis to see if we can find any trends in the dataset. Next, we will perform text preprocessing to convert textual data to numeric data that can be used by a machine learning algorithm. Finally, we will use machine learning algorithms to train and test our sentiment analysis models.

Result
======

I have create 5 file those are:

## [Sentiment Analysis Pre-precessing](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/1.%20Sentiment%20Analysis%20Pre-pocessing.ipynb)
1.	Importing the Required Libraries
2.	[Importing the Dataset](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/DATA/train_data_restaurant.tsv)
	1.	Check Null Values Present
	2.	Change the Label
	3.	Exploratory Data Analysis (EDA)
		1.	[Bar and pie plot for the label in dataset](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/image/label_ratio.png)
		2.	[WordCloud](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/image/wordcloud_all_data.png)
		3.	Text Preprocessing
		4.	Text Exploration
		5.	[WordCloud for Each Label](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/image/wordcloud_for_each_label.png)
		6.	The Most Common Word
			1.	[Positive reviews](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/image/most_common_word_in_positive_label.png)
			2.	[Negative reviews](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/image/most_common_word_in_negative_label.png)
		7.	The Less Common Word

## [Sentiment Analysis using Word to Number Representation methods](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/2.%20Sentiment%20Analysis%20using%20Word%20to%20Number%20Representation%20methods.ipynb)
1.	Importing the Required Libraries
2.	[Importing the Dataset](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/DATA/clean_text_train)
3.	Parameters
4.	Preparing the Matrix Layer
	1.	Word to Number Representation
	2.	Tokenizer
5.	Sentiment Analysis Models with simple Simple Neural Network
	1.	Early stopping to prevent overfitting
	2.	Evaluation
	3.	[Plot History](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/image/accuration_and_loss_plot_matrix_input.png)
6.	Comparing Word to Number Representation Methods
	1.	modes = binary
	2.	modes = count
	3.	modes = tfidf
	4.	modes = freq
	5.	[Box plot of summarizing the accuracy distributions per configuration](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/image/box_plot_of_summarizing_the_accuracy_distributions_per_configuration.png)
7.	Making a Prediction for New Reviews

## [Sentiment Analysis using Word2Vec (gensim)](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/3.%20Sentiment%20Analysis%20using%20Word2Vec%20(gensim).ipynb)
1.	Importing the Required Libraries
2.	[Importing the Dataset](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/DATA/clean_text_train)
3.	Preparing the Embedding Layer
4.	Create Word to Vector
5.	[Save the Model](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/model/model_w2v_skipgram.bin)
6.	Parameters
7.	Tokenizer
8.	Embedding
9.	Text Classification with CNN
	1.	[Plot Architecture](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/image/model_CNN.png)
	2.	Evaluation
	3.	Training
	4.	[Plot History](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/image/accuracy_and_loss_cnn_lstm_model.png)
10.	Text Classification with CNN-LSTM
	1.	[Plot Architecture](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/image/model_CNN_LSTM.png)
	2.	Evaluation
	3.	Training
	4.	[Plot History](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/image/accuracy_and_loss_cnn_model.png)
11.	Making a Prediction for New Reviews

## [experimental_word_to_vector_from_wikipedia_corpus](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/4.%20experimental_word_to_vector_from_wikipedia_corpus.ipynb)
1.	Importing the Required Libraries
2.	[Load the wikipedia file](https://dumps.wikimedia.org/idwiki/latest/idwiki-latest-pages-articles.xml.bz2)
3.	Create Word to Vector
4.	[Save the Model](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/model/wiki.id.word2vec.model)
5.	[Importing the Dataset](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/DATA/clean_text_train)
6.	Preparing the Embedding Layer
	1.	Tokenizer
	2.	Embedding
7.	Text Classification with CNN
	1.	[Plot Architecture](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/image/model_CNN_wiki.png)
	2.	Evaluation
	3.	Training
	4.	[Plot History](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/image/accuracy_and_loss_cnn_wiki_model.png)
8.	Text Classification with CNN-LSTM
	1.	[Plot Architecture](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/image/model_CNN_LSTM_wiki.png)
	2.	Evaluation
	3.	Training
	4.	[Plot History](https://gitlab.com/imanursar/nlp-ai-engineer/-/blob/master/image/accuracy_and_loss_cnn_wiki_model.png)
9.	Making a Prediction for New Reviews

## [Making a Prediction for Test dataset and New Reviews]

## [flask file]()
