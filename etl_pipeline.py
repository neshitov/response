import numpy as np
import pandas as pd
from sqlalchemy import create_engine

messages = pd.read_csv('./data/messages.csv')
categories = pd.read_csv('./data/categories.csv')
df = messages.merge(categories, on='id')

# split category column into dummy columns
category_colnames=[x[0:-2] for x in categories.iloc[0]['categories'].split(';')]
categories = df['categories'].str.split(';',expand=True)
categories.columns = category_colnames

# convert category column to numeric 1/0
for column in categories:
    categories[column] = categories[column].str[-1]
    categories[column] = pd.to_numeric(categories[column], errors='coerce')

df.drop(columns=['categories'], inplace=True)
df = pd.concat([df, categories], axis='columns')

# count duplicates
print(np.sum(df.duplicated()))
# drop duplicates
df.drop_duplicates(inplace=True)
# check duplicates
print(np.sum(df.duplicated()))

engine = create_engine('sqlite:///data/categorized_messages.db')
df.to_sql('messages', engine)
