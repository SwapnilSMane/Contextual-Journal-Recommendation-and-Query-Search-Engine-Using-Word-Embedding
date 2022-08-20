#!/usr/bin/env python
# coding: utf-8

# In[63]:


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
import re
from io import StringIO

stemmer = WordNetLemmatizer()
en_stop = set(nltk.corpus.stopwords.words('english'))



from mittens import GloVe, Mittens
from sklearn.feature_extraction import _stop_words
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import csv

import pandas as pd
import numpy as np

from gingerit.gingerit import GingerIt
# pip install tika
from tika import parser

import glob, os
import joblib


# In[65]:


class data_prepare:

    def __init__(self):
        print('import successfully')
        
    def clean_document(self, page):
        page = re.sub(r'[\t]',' ',page)
        page = re.sub(r'[\n]',' ',page)
        page = re.sub(r'[^.,a-zA-Z0-9 \n\.]',' ',page)
        page = re.sub(r'\s+',' ',page)

        return page


    #path = '/home/swapnil/Documents/ML1/Project/Implementation/community'


    def preprocess_text(self, document):
        document = re.sub(r'\d+', '', document)
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(document))
        document = re.sub(r'[^A-Za-z0-9]+',' ', document)

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)


        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        tokens = document.split()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if word not in en_stop]
        tokens = [word for word in tokens if len(word) > 3]

        preprocessed_text = ' '.join(tokens)

        return preprocessed_text

    def sentance_correction(self,corpus):
        textCorrected =[]
        for i in corpus:
            try:
                parser = GingerIt()
                text = parser.parse(i)['result']
                textCorrected.append(text)
            except:
                continue

        return textCorrected


    def extract_text(self, file):
        file_name = file #'ICDE_workshopv20.pdf'
        raw = parser.from_file(file_name)
        print(file)
        corpus = [self.clean_document(raw['content'])]
        corpus_doc = ' '.join(corpus)

        sentence = sent_tokenize(corpus_doc)
        corpus = [self.preprocess_text(i) for i in sentence]

        df = pd.DataFrame()
        df['text'] = [' '.join(corpus)]
        #df['correct_text'] = ' '.join(textCorrected)
        df['paper_title'] = [file_name.split('.')[0]]
        temp = df
        return temp


# In[ ]:




