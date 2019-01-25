import pandas as pd
import pickle
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import json
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import sys, os
import argparse

argparser = argparse.ArgumentParser(description='Training a messages classfier')
argparser.add_argument('database_path', help='Path to database')
argparser.add_argument('model_path', help='Path to model file')
args = argparser.parse_args()

os.system("python -m spacy download en")
nlp = spacy.load('en')
engine = create_engine(os.path.join('sqlite:///',args.database_path))


df = pd.read_sql_table('messages', engine)
#df = df.iloc[0:30]
X = df['message']

cat_columns = df.columns[5:]
#cat_columns = [col for col in cat_columns if df[col].nunique()>1]
#cat_columns = [cat_columns[i] for i in [0,2]]
#print(df[cat_columns])
Y = df[cat_columns].values
#print(Y.shape)
#sys.exit()
#Y = df[cat_columns].values

#Extracting named entities
def entities(text):
    doc = nlp(text)
    out = ""
    for ent in doc.ents:
        out = out + ent.text.lower() + ' '
    return out

class Entitizer:
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
        VotingClassifier(estimators=[('svm', SVC(gamma=0.6, probability = True)),
                                     ('NB',  ComplementNB())],
                         voting='soft', weights=[1, 1])))
    ])


'''
'clf__estimator__svm__kernel': 'rbf'
'clf__estimator__weights': [1, 1]
'clf__estimator__svm__degree'
'clf__estimator__svm__gamma'
print(voting_pipeline.get_params())
'''

def weighted_test_f1(estimator, X_test, Y_test):
    predict_test = estimator.predict(X_test)
    score = 0
    for i in range(Y_test.shape[1]):
        score += classification_report(Y_test[:,i], predict_test[:,i],
                                       output_dict=True)['weighted avg']['f1-score']
    return score/(Y_test.shape[1])

gs = GridSearchCV(voting_pipeline, param_grid={
                'clf__estimator__svm__kernel': ['rbf'],
                'clf__estimator__weights': [[1, 1],[0,1],[1,0]],
                'clf__estimator__svm__degree': [3],
                'clf__estimator__svm__gamma': [0.6]
                }, scoring=weighted_test_f1, cv=2)

gs.fit(X,Y)

''' find the best parameters '''
model = gs.best_estimator_

''' evaluate the best model fit it on the train data and compute statistics on test data'''
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
model.fit(X_train, Y_train)

predict_test = model.predict(X_test)
for i in range(Y_test.shape[1]):
    print('column '+str(i)+':')
    print(classification_report(Y_test[:,i], predict_test[:,i],
                                output_dict=True)['weighted avg']['f1-score'])

with open(args.model_path, 'wb') as file:
    pickle.dump(model, file)

print('after loading')

with open(args.model_path, 'rb') as file:
    loaded_model = pickle.load(file)

predict_test = loaded_model.predict(X_test)
for i in range(Y_test.shape[1]):
    print('column '+str(i)+':')
    print(classification_report(Y_test[:,i], predict_test[:,i],
                                output_dict=True)['weighted avg']['f1-score'])


sys.exit()



results = pd.DataFrame({'model_name':[],'weighted_test_f1_score':[]})
results = pd.read_csv('training_metrics.csv')
pipeline = svm_pipeline
name = 'svm'
parameters = {'clf__gamma':[0.65], 'tfidf__use_idf': [False,True]}
'''
for pars in ParameterGrid(parameters):
    pipeline.set_params(**pars)
    pipeline.fit(X_train, Y_train[:,0])
    score = weighted_test_f1(pipeline, X_test, Y_test)
    model_name = name
    for k in parameters.keys():
        model_name = model_name + '_' + str(k) + '_' + str(pars[k])
    print(model_name, score)
    results = results.append({'model_name': model_name, 'weighted_f1_score': score}, ignore_index = True)
'''

enhanced_svm_pipeline.fit(X_train, Y_train[:,0])
predict_test = enhanced_svm_pipeline.predict(X_test)
predict_train = enhanced_svm_pipeline.predict(X_train)
results = {}

print('what we get')
print('train:')
results['svm']={}
results['svm']['train'] = classification_report(Y_train[:,0], predict_train)
print(results['svm']['train'])
print('test:')
results['svm']['test'] = classification_report(Y_test[:,0], predict_test)
print(results['svm']['test'])
with open('result.json', 'w') as fp:
    json.dump(results, fp)

sys.exit()

''' Test model '''
for i in range(Y.shape[1]):
    print(cat_columns[i])
    print(classification_report(Y_test[:,i], predict_test[:,i]))

''' Grid search for paramter tuning. We will compare models by the accuracy of
classifying if the message is realted to the disaster'''

def scorer(estimator,X,Y):
    pred = estimator.predict(X)
    return f1_score(Y[:,0], pred[:,0])

gs = GridSearchCV(pipeline, param_grid={'clf__estimator__n_estimators':[100,500],
                             'clf__estimator__max_depth':[3,5,10]
                              }, scoring=scorer)
gs.fit(X,Y)
results = pd.DataFrame.from_dict(gs.cv_results_)
results.to_csv('training_metrics.csv')
# return best parameters:
print(gs.best_params_)

''' Test the optimal model'''

estimator = gs.best_estimator_
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

estimator.fit(X_train, Y_train)
predict_test = estimator.predict(X_test)

for i in range(Y.shape[1]):
    print(cat_columns[i])
    print(classification_report(Y_test[:,i], predict_test[:,i]))

''' Try to improve model further: Try to use naive bayes classifier'''

naive_bayes_pipeline = Pipeline([
        ('vectorize', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(GaussianNB()))
    ])

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
naive_bayes_pipeline.fit(X_train,Y_train)
predict_test = naive_bayes_pipeline.predict(X_test)

for i in range(Y.shape[1]):
    print(cat_columns[i])
    print(classification_report(Y_test[:,i], predict_test[:,i]))
