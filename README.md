# Disaster response project

This project builds a model that analyses and categorizes disaster messages. The web app build on this model is available at
[https://arcane-reef-23233.herokuapp.com/](https://arcane-reef-23233.herokuapp.com/)

## Dataset 
The dataset consists of around 30 thousand real messages sent during disaster events,
with binary labels indicating if a message fits into one of the 36 categories (if message is related to the disaster, 
if a message is a request, if it concerns weather conditions, etc.)

## Model
The model uses [NLTK](https://www.nltk.org/) and [spacy](https://spacy.io/) libraries for tokenization. It uses 
tf-idf vectorization and extracts named entities from the text as additional input for the model. As a classifier 
we use sklearn Complement Naive-Bayes classifier due to its training speed and its suitability fo heavily imbalanced
data sets.

## Files
**process_data.py** performs the data cleaning and saves messages with labels to a sqlite database
**response_model.py** contains the ResponseModel class and implements the tokenization and ML pipelines, 
implements training and performance estimation.
**train_classifier.py** connects to the messages database, trains a model and saves it to the file
**/local_app/app.py** is a dash plotly web app with visualizations and online messages classification

## Model accuracy
The Model has the follwoing F1-score averaged over label 0 and 1
![f1 barplot](https://github.com/neshitov/response/blob/master/model_f1_barplot.png)
