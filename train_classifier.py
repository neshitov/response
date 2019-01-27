''' Training a model for message classifier
    Usage: train_classifier.py /path_to_database /path_to_store_the_model
'''
import pandas as pd
import os
import argparse
import time
import joblib
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from response_model import ResponseModel


argparser = argparse.ArgumentParser(description='Training a messages classfier')
argparser.add_argument('database_path', help='Path to database')
argparser.add_argument('model_path', help='Path to model file')
args = argparser.parse_args()

''' read the data '''
engine = create_engine(os.path.join('sqlite:///', args.database_path))
df = pd.read_sql_table('messages', engine)
X = df['message']
cat_columns = df.columns[5:]
cat_columns = cat_columns[0:2]
Y = df[cat_columns]

''' Initialize the model'''
model = ResponseModel()

''' Training'''
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
start = time.time()
model.train(X_train, Y_train)
print('trained in ',time.time()-start)

''' save model performance'''
results = model.f1_performance(X_test, Y_test)
print(results)
results.to_csv('model_f1_performance.csv')

''' save model to file '''
with open(args.model_path, 'wb') as file:
    joblib.dump(model, file)
