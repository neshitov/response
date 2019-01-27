import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from response_model import tokenize
from collections import Counter
import pandas as pd
from sqlalchemy import create_engine
import os
import numpy as np
import pickle

''' Create wordcloud from precomputedd word frequencies from database messages '''

with open('word_count.pkl', 'rb') as file:
    X = pickle.load(file)

wordcloud = WordCloud(background_color="white", width=480, height=480, margin=0).generate_from_frequencies(X)
wordcloud.to_file('cloud.png')
