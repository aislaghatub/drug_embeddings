# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:11:09 2020

@author: aosulli4
"""

import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt

from gensim.test.utils import datapath
from gensim import utils
from gensim.models import Word2Vec

import gensim.models
from sklearn.manifold import TSNE

# remove punctuation
def remove_punctuation(s):
    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
    return s

# get cosine similarity
def cosine_distance (model, word,target_list , num) :
    cosine_dict ={}
    word_list = []
    a = model.wv[word]
    for item in target_list :
        if item != word :
            b = model.wv[item]
            cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
            cosine_dict[item] = cos_sim
    dist_sort=sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) ## in Descedning order 
    for item in dist_sort:
        word_list.append((item[0], item[1]))
    return word_list[0:num]

def display_closestwords_tsnescatterplot(model, word, topn, size,approved_drugs, other_drugs,drug_stages):
    
    arr = np.empty((0,size), dtype='f')
    word_labels = [word]
    close_words = model.wv.similar_by_word(word,topn)
    arr = np.append(arr, np.array([model.wv[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model.wv[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        if label in approved_drugs:
           plt.text(x, y, label, bbox=dict(facecolor='red', alpha=0.5))
        elif label in other_drugs:
            label_stage = drug_stages[other_drugs.index(label)] # find index of the drug to return its stage in trials
            if label_stage == 0:
                plt.text(x, y, label, bbox=dict(facecolor='blue', alpha=0.5))
            elif label_stage == 1:
                plt.text(x, y, label, bbox=dict(facecolor='cyan', alpha=0.5))
            elif label_stage == 2:
                plt.text(x, y, label, bbox=dict(facecolor='green', alpha=0.5))
            elif label_stage == 3:
                plt.text(x, y, label, bbox=dict(facecolor='yellow', alpha=0.5))
            elif label_stage == 4:
                plt.text(x, y, label, bbox=dict(facecolor='magenta', alpha=0.5))
            
    plt.xlim(x_coords.min()+0.5, x_coords.max()+0.5)
    plt.ylim(y_coords.min()+0.5, y_coords.max()+0.5)
    plt.show()
    
# # import and clean the abstract data
# df=pd.read_pickle("abstracts.pkl") # import data
# df=df.dropna() # drop rows that have nan value 
# df=df[~df.text.str.contains("nan")] #or have 'nan' string in them
# df['text']=df['text'].str.lower() # make all text lower case
# df['text']=df['text'].apply(remove_punctuation) # try applying it just to one row

# text_list = [row.split(' ') for row in df['text']] # convert into list

# remove title words... maybe later
# title_words=['abstract','background','rationale','aims','purpose']

# model = gensim.models.Word2Vec(text_list, size=100, window=5, min_count=5, workers=4, sg=1)
# model.save('w2v_abs_model.model')


# for loading later...
model = Word2Vec.load("w2v_abs_model.model")

approved_drugs = ['ipilimumab', 'nivolumab','pembrolizumab','atezolizumab','avelumab','durvalumab','cemiplimab']
# check what words are similar to approved drug names
top_twenty = model.wv.most_similar(approved_drugs[0],[],20)


# # # Show the most similar drugs to 'cancer' by cosine distance 
# cosine_distance (model,'cancer',approved_drugs[:5],5)
# model.wv.similarity('cancer',approved_drugs[0]) # gensim method that calculates cosine similarity between 2 entities

df_drug = pd.read_csv('drugs.csv')
other_drugs=df_drug['Drug Name'].tolist() # convert series to list
drug_stages=df_drug['Stage'].tolist() # convert series to list

num_sim_words=150
size=100
display_closestwords_tsnescatterplot(model, approved_drugs[0], num_sim_words, size,approved_drugs,other_drugs,drug_stages) 




