''' Describing model '''

import pandas as pd
import pickle
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
import spacy
import os
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
os.system("python -m spacy download en")
nlp = spacy.load('en')

def entities(text):
    '''returns string of found named entities'''
    doc = nlp(text)
    out = ""
    for ent in doc.ents:
        out = out + ent.text.lower() + ' '
    return out

class Entitizer:
    ''' transformer extracting named entities'''
    def fit(self, X, y):
        return self
    def transform(self, X):
        return X.apply(entities)

def tokenize(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token).strip() for token in tokens]
    return tokens

''' pipeline combining Naive Bayes and k-neighbors classifiers'''
voting_pipeline = Pipeline([
    ('process', FeatureUnion([
    ('all_tokens', Pipeline([
            ('vectorize', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())])),
    ('named_entities', Pipeline([
            ('named_entities', Entitizer()),
            ('vectorize', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())]))
                ])),
    ('clf', MultiOutputClassifier(
                                VotingClassifier(estimators=[('rf', KNeighborsClassifier(n_neighbors=3)),
                                         ('knn', KNeighborsClassifier(n_neighbors=5)),
                                         ('NB',  ComplementNB())],
                             voting='hard', weights=[1, 1, 1]),
                               n_jobs = -1))
    ])


def weighted_test_f1(estimator, X_test, Y_test):
    predict_test = estimator.predict(X_test)
    score = 0
    for i in range(Y_test.shape[1]):
        score += classification_report(Y_test[:,i], predict_test[:,i], output_dict=True)['weighted avg']['f1-score']
    return score

def train_classifier(X, Y):
    ''' function to train classifer
        Args:
              X, Y: messages and labels
        Returns:
              model: classifier for all categorical columns.


    '''
    gs = GridSearchCV(voting_pipeline, param_grid={
                    #'clf__svm__kernel': ['rbf'],
                    'clf__estimator__weights': [[1, 1, 1],[0,1,0],[0,1,1]],
                    #'clf__svm__degree': [3],
                    #'clf__svm__gamma': [0.6]
                    }, scoring=weighted_test_f1, cv=2)
    gs.fit(X,Y.values)
    ''' find the best parameters '''
    model = gs.best_estimator_
    return model


class ResponseModel():
    ''' Class to store classifiers for all categorical columnsself.
        Implements training, prediction and performance evaluation.
    '''
    classifiers = None
    cat_columns = None
    def train(self, X, Y):
        clf = train_classifier(X, Y)
        self.classifier = clf
        self.cat_columns = list(Y.columns)

    def predict(self, X):
        assert self.classifier is not None, 'No classifier available yet. Call the train method first'
        return pd.DataFrame(dict([(self.cat_columns[i], self.classifier.predict(X)[:,i]) for i in range(len(self.cat_columns))]))

    def f1_performance(self, X_test, Y_test):
        predict_test = self.predict(X_test)
        indices = ['0', '1', 'micro avg', 'macro avg', 'weighted avg']
        f1_results = pd.DataFrame(dict([(col,[np.nan,np.nan,np.nan,np.nan,np.nan])
                                        for col in self.cat_columns]), index = indices)
        for col in self.cat_columns:
            report = classification_report(Y_test[col], predict_test[col],output_dict=True)
            for index in indices:
                if index in report.keys():
                    f1_results.loc[index, col] = report[index]['f1-score']
        return f1_results
