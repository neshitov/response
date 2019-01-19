import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import SVC
import json
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys

engine = create_engine('sqlite:///data/categorized_messages.db')
df = pd.read_sql_table('messages', engine)
df = df.iloc[0:300,:]
#print(df.columns)
#print(len(df.columns))

X = df['message']

cat_columns = df.columns[5:]
Y = df[cat_columns].values



def tokenize(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token).strip() for token in tokens]
    return tokens


#pipeline = Pipeline([
#        ('vectorize', CountVectorizer(tokenizer=tokenize)),
#        ('tfidf', TfidfTransformer()),
#        ('clf', MultiOutputClassifier(RandomForestClassifier(max_depth=5, n_estimators=10)))
#    ])


single_pipeline = Pipeline([
        ('vectorize', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', ComplementNB())
    ])

svm_pipeline = Pipeline([
        ('vectorize', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', SVC(gamma=0.6))
    ])


def weighted_test_f1(pipe, X_test,Y_test):
    predict_test = pipe.predict(X_test)
    return classification_report(Y_test[:,0],
            predict_test,output_dict=True)['weighted avg']['f1-score']


''' Train pipeline '''
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

results = pd.DataFrame({'model_name':[],'weighted_test_f1_score':[]})
results = pd.read_csv('training_metrics.csv')
pipeline = svm_pipeline
name = 'svm'
parameters = {'clf__gamma':[0.65], 'tfidf__use_idf': [False,True]}

for pars in ParameterGrid(parameters):
    pipeline.set_params(**pars)
    pipeline.fit(X_train, Y_train[:,0])
    score = weighted_test_f1(pipeline, X_test, Y_test)
    model_name = name
    for k in parameters.keys():
        model_name = model_name + '_' + str(k) + '_' + str(pars[k])
    print(model_name, score)
    results = results.append({'model_name': model_name, 'weighted_f1_score': score}, ignore_index = True)

print(results)
results.to_csv('training_metrics.csv')
sys.exit()

svm_pipeline.fit(X_train, Y_train[:,0])
predict_test = svm_pipeline.predict(X_test)
predict_train = svm_pipeline.predict(X_train)
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
