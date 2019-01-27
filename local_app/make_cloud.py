import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from response_model import tokenize
from collections import Counter
import pandas as pd
from sqlalchemy import create_engine
import os

engine = create_engine(os.path.join('sqlite:///', './data/categorized_messages.db'))
df = pd.read_sql_table('messages', engine)
X = df['message']
X = X.apply(tokenize)
print (X)
'''
text=("Python Python Python Matplotlib Matplotlib Seaborn Network Plot Violin Chart Pandas Datascience Wordcloud Spider Radar Parrallel Alpha Color Brewer Density Scatter Barplot Barplot Boxplot Violinplot Treemap Stacked Area Chart Chart Visualization Dataviz Donut Pie Time-Series Wordcloud Wordcloud Sankey Bubble")
 # Create the wordcloud object
wordcloud = WordCloud(width=480, height=480, margin=0).generate(text)
wordcloud.to_file('cloud.png')
'''
