
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:45:58 2020

@author: aosulli4

Reads in the abstracts, removes stopwords, finds frequency of words in abstracts 
for approved drugs and abstracts for developing drugs. 
Plots wordcloud of the words in abstracts for the approved and developing drugs seperately. 


"""

import pickle
import pandas as pd 
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk import FreqDist
from wordcloud import WordCloud


with open('abs_fiveTofifty_minus1yr.data', 'rb') as filehandle:
    # read the data as binary data stream
    abstractsList = pickle.load(filehandle)
    
# remove stopwords
filt_absList=[]
stop_words = set(stopwords.words('english')) 

for i in range(len(abstractsList)):
    filtered_abs = [w for w in abstractsList[i] if not w in stop_words] 
    filt_absList.append(filtered_abs)
    
approved_drugs = ['ipilimumab', 'nivolumab','pembrolizumab','atezolizumab','avelumab','durvalumab','cemiplimab']

df_drug = pd.read_csv('drugs.csv')
other_drugs=df_drug['Drug Name'].tolist() # convert series to list
drug_stages=df_drug['Stage'].tolist() # convert series to list

# find most common words in abstracts of approved drugs vs not approved drugs
for i in range(len(approved_drugs)):
    for j in range(len(filt_absList)):
        if approved_drugs[i] in filt_absList[j]:
            freq=FreqDist(filt_absList[j])
            app_freq_df = pd.DataFrame(list(freq.items()), columns = ["Word","Frequency"])
              
for i in range(len(other_drugs)):
    for j in range(len(filt_absList)):
        if other_drugs[i] in filt_absList[j]:
            freq=FreqDist(filt_absList[j])
            other_freq_df = pd.DataFrame(list(freq.items()), columns = ["Word","Frequency"])
              

# sort in order of frequency, most common first
app_freq_df.sort_values(by=['Frequency'],ascending=False,inplace=True)
other_freq_df.sort_values(by=['Frequency'],ascending=False,inplace=True)

    
## -------- Wordcloud of word frequency ---------------------
d = {}
for a, x in app_freq_df.values:
    d[a] = x


wordcloud = WordCloud()
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear",cmap='RdBu')
plt.axis("off")
plt.title('approved drugs')
plt.show()


d = {}
for a, x in other_freq_df.values:
    d[a] = x


wordcloud = WordCloud()
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title('other drugs')
plt.show()

