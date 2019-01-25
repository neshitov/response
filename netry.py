'''
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')

ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent
#print(preprocess(ex))
print(nltk.help.upenn_tagset('JJ'))
'''

import spacy
nlp = spacy.load('en')
doc = nlp('Apple is looking at buying U.K. startup for $1 billion in Los Angeles')
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

def entities(text):
    doc = nlp(text)
    out = ""
    for ent in doc.ents:
        out = out + ent.text.lower() + ' '
    return out
a = 1
b = 2
c = 3
list = [a,b,c]
for x in list:
    print(x)
#print(entities('We have a lot of problem in Port-de-Paix'))
