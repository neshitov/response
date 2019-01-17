import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys

engine = create_engine('sqlite:///data/categorized_messages.db')
df = pd.read_sql_table('messages', engine)
#df = df.iloc[0:200,:]
#print(df.columns)
#print(len(df.columns))

X = df['message']
Y = df[df.columns[5:]].values

def tokenize(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token).strip() for token in tokens]
    return tokens


pipeline = Pipeline([
        ('vectorize', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10,max_depth=3)))
    ])

#print(pipeline.get_params().keys())
#sys.exit()
def scorer(estimator,X,Y):
    pred = estimator.predict(X)
    return accuracy_score(Y[:,0], pred[:,0])

gs = GridSearchCV(pipeline, {'clf__estimator__n_estimators':[10,50,100,200],
                             'clf__estimator__max_depth':[3,4,5,6]
                              }, scoring=scorer, cv=2)
gs.fit(X,Y)
results = pd.DataFrame.from_dict(gs.cv_results_)
results.to_csv('training_metrics.csv')
print(results[results.columns[5:9]])

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
#print(Y_train['related'].values)

#pipeline.fit(X_train, Y_train)
#pred_train = pipeline.predict(X_train)
#pred_test = pipeline.predict(X_test)

#print('train accuracy',accuracy_score(Y_train[:,0], pred_train[:,0]))
#print('test accuracy',accuracy_score(Y_test[:,0], pred_test[:,0]))
