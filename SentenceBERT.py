#!/usr/bin/env python
# coding: utf-8

# ## Embeddings by SentenceBERT

# In[55]:


# https://www.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/
# glove https://nlp.stanford.edu/pubs/glove.pdf

from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


# In[71]:


from sentence_transformers import SentenceTransformer
class Bert_embedding:
    
    def __init__(self, df_sentances):
        self.sentances = list(df_sentances['sentances']) # cleaned and corrected sentances
        self.query = None # cleaned and corrected query
        self.df_sentances = df_sentances
        self.sentence_embeddings = None
        self.sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        
        
    def cosine_similarity(self, a, b):
        nominator = np.dot(a, b)

        a_norm = np.sqrt(np.sum(a**2))
        b_norm = np.sqrt(np.sum(b**2))

        denominator = a_norm * b_norm

        cosine_similarity = nominator / denominator

        return cosine_similarity
    
    def query_embedding(self, query, sbert_model):
        query_vector= sbert_model.encode(query)

        return query_vector

    
    
    def fit(self):
        self.sentence_embeddings = self.sbert_model.encode(self.sentances)
    
    
    def fit_transform(self, query):
        self.query = query
        query_vector = self.query_embedding(self.query, self.sbert_model)

        cosine_score = []
        for i in self.sentence_embeddings:
            cosine_score.append(self.cosine_similarity(query_vector,i))


        self.df_sentances['cosine_score'] = cosine_score

        self.df_sentances = self.df_sentances.sort_values(by = 'cosine_score',ascending = False)
        self.df_sentances.reset_index(inplace=True,drop=True)
    
        return self.df_sentances


# In[ ]:




