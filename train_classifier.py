''' Training a model for message classifier '''

import pandas as pd
import pickle
import numpy as np
from sqlalchemy import create_engine
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
import spacy
import os
import argparse
import time
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


argparser = argparse.ArgumentParser(description='Training a messages classfier')
argparser.add_argument('database_path', help='Path to database')
argparser.add_argument('model_path', help='Path to model file')
args = argparser.parse_args()

os.system("python -m spacy download en")
nlp = spacy.load('en')
engine = create_engine(os.path.join('sqlite:///', args.database_path))

df = pd.read_sql_table('messages', engine)
#df = df.iloc[0:10]
X = df['message']

cat_columns = df.columns[5:]
Y = df[cat_columns]

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

''' pipeline combining Naive Bayes with support vector machines'''
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
    ('clf', VotingClassifier(estimators=[('svm', LinearSVC()),
                                         ('knn', KNeighborsClassifier()),
                                         ('NB',  ComplementNB())],
                             voting='hard', weights=[1, 1]))
    ])

''' naive Bayes classifier pipeline'''
NB_pipeline = Pipeline([
    ('process', FeatureUnion([
    ('all_tokens', Pipeline([
            ('vectorize', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())])),
    ('named_entities', Pipeline([
            ('named_entities', Entitizer()),
            ('vectorize', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())]))
                ])),
    ('clf', ComplementNB())
    ])

def weighted_test_f1(estimator, X_test, Y_test_column):
    predict_test = estimator.predict(X_test)
    return classification_report(Y_test_column, predict_test, output_dict=True)['weighted avg']['f1-score']

def train_classifier(column, X, Y):
    ''' function to train categorical classifer for each categorical column
        Args:
              column: categorical column name
              X, Y: messages and labels
        Returns:
              column_model: classifier for the given column label.

        Voting of support vector classifier and naive Bayes is used. For heavily unbalanced data
        the Naive Bayes classifier is used to avoid error when one attempts to fit support vector classifier
        to sample with no negative examples.
    '''
    if (Y[column].nunique()>1) and (np.min(Y[column].value_counts(normalize=True).values) > 0.04):
        gs = GridSearchCV(voting_pipeline, param_grid={
                    #'clf__svm__kernel': ['rbf'],
                    'clf__weights': [[1, 1]],
                    #'clf__svm__degree': [3],
                    #'clf__svm__gamma': [0.6]
                    }, scoring=weighted_test_f1, cv=2)
        gs.fit(X,Y[column].values)
        ''' find the best parameters '''
        column_model = gs.best_estimator_
    else:
        gs = GridSearchCV(NB_pipeline, param_grid={
                    'process__all_tokens__tfidf__use_idf': [True,False]
                    }, scoring=weighted_test_f1, cv=2)
        gs.fit(X,Y[column].values)
        ''' find the best parameters '''
        column_model = gs.best_estimator_
    return column_model


class ResponseModel():
    ''' Class to store classifiers for all categorical columnsself.
        Implements training, prediction and performance evaluation.
    '''
    classifiers = {}
    def train(self, X, Y):
        for column in cat_columns:
            clf = train_classifier(column, X, Y)
            self.classifiers[column] = clf

    def predict(self, X):
        assert self.classifiers, 'No classifier available yet. Call the train method first'
        return pd.DataFrame(dict([(col, self.classifiers[col].predict(X)) for col in cat_columns]))

    def f1_performance(self, X_test, Y_test):
        predict_test = self.predict(X_test)
        indices = ['0', '1', 'micro avg', 'macro avg', 'weighted avg']
        f1_results = pd.DataFrame(dict([(col,[np.nan,np.nan,np.nan,np.nan,np.nan])
                                        for col in cat_columns]), index = indices)
        for col in cat_columns:
            report = classification_report(Y_test[col], predict_test[col],output_dict=True)
            for index in indices:
                if index in report.keys():
                    f1_results.loc[index, col] = report[index]['f1-score']
        return f1_results


''' Initialize the model'''
model = ResponseModel()

''' Training'''
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
model.train(X_train, Y_train)

'''save model performance'''
results = model.f1_performance(X_test, Y_test)
print(results)
results.to_csv('model_f1_performance.csv')

''' save model to file '''
with open(args.model_path, 'wb') as file:
    pickle.dump(model, file)
