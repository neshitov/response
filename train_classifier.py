''' Training a model for message classifier '''
import pandas as pd
import pickle
import os
import argparse

from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from response_model import ResponseModel


argparser = argparse.ArgumentParser(description='Training a messages classfier')
argparser.add_argument('database_path', help='Path to database')
argparser.add_argument('model_path', help='Path to model file')
args = argparser.parse_args()

engine = create_engine(os.path.join('sqlite:///', args.database_path))

df = pd.read_sql_table('messages', engine)
#df = df.iloc[0:100]
X = df['message']

cat_columns = df.columns[5:]
#cat_columns = cat_columns[0:2]
Y = df[cat_columns]

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

with open(args.model_path, 'rb') as fp:
     loaded_model = pickle.load(fp)
