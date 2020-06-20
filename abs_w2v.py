# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:11:09 2020

@author: aosulli4

Reads in the abstracts, removes stopwords, trains word2vec model, saves model 
Plots t-SNE scatterplot of cosine similarity between an approved drug and all other drugs. 

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from nltk.corpus import stopwords 
import gensim.models
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import pickle
     
text_size = 16
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
    
    plt.figure()
    # plt.scatter(x_coords, y_coords,5,'gray',alpha=0.3) # to plot a scatter dot for every word
    colors = [ cm.YlOrRd(x) for x in np.linspace(0, 1, 6) ] # split colormap into 6
    for label, x, y in zip(word_labels, x_coords, y_coords):
        # plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points') # to plot every word in text
        if label in approved_drugs:
           plt.text(x, y, label, bbox=dict(facecolor=colors[5], alpha=0.5),fontsize=text_size)
        elif label in other_drugs:
            label_stage = drug_stages[other_drugs.index(label)] # find index of the drug to return its stage in trials
            if label_stage == 0:
                plt.text(x, y, label, bbox=dict(facecolor=colors[int(label_stage)], alpha=0.5),fontsize=text_size)
            elif label_stage == 1:
                plt.text(x, y, label, bbox=dict(facecolor=colors[int(label_stage)], alpha=0.5),fontsize=text_size)
            elif label_stage == 2:
                plt.text(x, y, label, bbox=dict(facecolor=colors[int(label_stage)], alpha=0.5),fontsize=text_size)
            elif label_stage == 3:
                plt.text(x, y, label, bbox=dict(facecolor=colors[int(label_stage)], alpha=0.5),fontsize=text_size)
            elif label_stage == 4:
                plt.text(x, y, label, bbox=dict(facecolor=colors[int(label_stage)], alpha=0.5),fontsize=text_size)
    

    plt.xlim(x_coords.min()+0.5, x_coords.max()+0.5)
    plt.ylim(y_coords.min()+0.5, y_coords.max()+0.5)
    plt.show()



# train model for drugs with at least 5 abstracts and use max of 50 abstracts
# to help make comparisons between drugs fair

with open('abs_fiveTo1k_minus1yr.data', 'rb') as filehandle:
    # read the data as binary data stream
    abstractsList = pickle.load(filehandle)
    
# remove stopwords
filt_absList=[]
stop_words = set(stopwords.words('english')) 

for i in range(len(abstractsList)):
    filtered_abs = [w for w in abstractsList[i] if not w in stop_words] 
    filt_absList.append(filtered_abs)


    
#----------------------------Train Word2Vec model----------------------------------

model = gensim.models.Word2Vec(filt_absList, size=100, window=5, min_count=5, workers=4, sg=1)
model.save('abs_fiveTo1k_minus1yr.model')



#-----------------------------Plot 

# for loading later...
model = Word2Vec.load("abs_fiveTofifty_minus1yr.model")


approved_drugs = ['ipilimumab', 'nivolumab','pembrolizumab','atezolizumab','avelumab','durvalumab','cemiplimab']

df_drug = pd.read_csv('drugs.csv')
other_drugs=df_drug['Drug Name'].tolist() # convert series to list
drug_stages=df_drug['Stage'].tolist() # convert series to list
         
num_sim_words=len(model.wv.vocab)
size=100
# for i in range(len(approved_drugs)):
    # plt.figure()
    # if approved_drugs[i] in model.wv.vocab:
        # display_closestwords_tsnescatterplot(model, approved_drugs[i], num_sim_words, size,approved_drugs,other_drugs,drug_stages) 

        
display_closestwords_tsnescatterplot(model, approved_drugs[0], num_sim_words, size,approved_drugs,other_drugs,drug_stages) 


# positive_health_words =['recovery','cure','improve'] 
# negative_health_words =['toxic','fatal','deterioration'] 

# for j in range(len(positive_health_words)):
#     display_closestwords_tsnescatterplot(model, positive_health_words[j], num_sim_words, size,approved_drugs,other_drugs,drug_stages) 
    
#     display_closestwords_tsnescatterplot(model, negative_health_words[j], num_sim_words, size,approved_drugs,other_drugs,drug_stages) 


