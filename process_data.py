import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import argparse
import os

argparser = argparse.ArgumentParser(description='ETL Pipeline for messages and categories')
argparser.add_argument('messages_path', help='Path to messages csv file')
argparser.add_argument('categories_path', help='Path to categories csv file')
argparser.add_argument('database_path', help='Path to resulting database')

args = argparser.parse_args()

messages = pd.read_csv(args.messages_path)
categories = pd.read_csv(args.categories_path)
df = messages.merge(categories, on='id')

# split category column into dummy columns
category_colnames=[x[0:-2] for x in categories.iloc[0]['categories'].split(';')]
categories = df['categories'].str.split(';',expand=True)
categories.columns = category_colnames

# convert category column to numeric 1/0
for column in categories:
    categories[column] = categories[column].str[-1]
    categories[column] = pd.to_numeric(categories[column], errors='coerce')
    categories[column] = (categories[column] > 0).astype('int32')

df.drop(columns=['categories'], inplace=True)
df = pd.concat([df, categories], axis='columns')

# drop duplicates
df.drop_duplicates(inplace=True)

engine = create_engine(os.path.join('sqlite:///',args.database_path))
df.to_sql('messages', engine)
