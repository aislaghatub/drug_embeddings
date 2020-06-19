# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:04:40 2020

@author: aosulli4
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm

from gensim.models import Word2Vec
from sklearn.decomposition import PCA as sPCA
from sklearn import manifold #MSD, t-SNE
    
model = Word2Vec.load("abs_fiveTofifty_minus1yr.model")

# contains the list of all unique words in pre-trained word2vec vectors
w2v_vocabulary = model.wv.vocab
w2v_words = list(w2v_vocabulary.keys()) 

wvec=[]
for wd in range(len(w2v_words)):
    wvec.append(model.wv.word_vec(w2v_words[wd]))


spca = sPCA(n_components=2)
coords = spca.fit_transform(wvec)    
 
# tsne = manifold.TSNE(n_components=2)
# coords = tsne.fit_transform(wvec)

approved_drugs = ['ipilimumab', 'nivolumab','pembrolizumab','atezolizumab','avelumab','durvalumab','cemiplimab']

df_drug = pd.read_csv('drugs.csv')
other_drugs=df_drug['Drug Name'].tolist() # convert series to list
drug_stages=df_drug['Stage'].tolist() # convert series to list


text_size=12
plt.figure()

x_coords=coords[:,0]
y_coords=coords[:,1]
# plt.scatter(x_coords, y_coords,5,'gray',alpha=0.3)
colors = [ cm.YlOrRd(x) for x in np.linspace(0, 1, 6) ] # split colormap into 6
lim = max([abs(x) for x in coords[:,0] + coords[:,1]])
plt.xlim([-lim,lim])
plt.ylim([-lim,lim])
for label, x, y in zip(w2v_words, x_coords, y_coords):
    # plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    if label in approved_drugs:
       plt.text(x, y, label, bbox=dict(facecolor=colors[5], alpha=0.5),fontsize=text_size)
    elif label in other_drugs:
        label_stage = drug_stages[other_drugs.index(label)] # find index of the drug to return its stage in trials
        if label_stage == 0:
            plt.text(x, y, label, bbox=dict(facecolor=colors[0], alpha=0.5),fontsize=text_size)
        elif label_stage == 1:
            plt.text(x, y, label, bbox=dict(facecolor=colors[1], alpha=0.5),fontsize=text_size)
        elif label_stage == 2:
            plt.text(x, y, label, bbox=dict(facecolor=colors[2], alpha=0.5),fontsize=text_size)
        elif label_stage == 3:
            plt.text(x, y, label, bbox=dict(facecolor=colors[3], alpha=0.5),fontsize=text_size)
        elif label_stage == 4:
            plt.text(x, y, label, bbox=dict(facecolor=colors[4], alpha=0.5),fontsize=text_size)
                
            

