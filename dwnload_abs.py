# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:46:37 2020

@author: aosulli4

Downloads abstracts from pubmed using the drug names in drugs.csv as search terms 
and the trial date for setting the max date of the search. 
Prepares the abstarcts for word2vec.
Saves the abstracts as list of words. Also saves the number of abstracts per drug.

"""

# using biophython
from Bio import Entrez
from Bio import Medline
from tqdm import tqdm # progress meter for abstract download

import pandas as pd
import string
import pickle

# Change this email to your email address
#Entrez.email = "aosulli4@ur.rochester.edu"

# remove punctuation
def remove_punctuation(s):
    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
    return s


df_drug = pd.read_csv('drugs.csv')
df_drug = df_drug.reset_index()

minyear=1600 # need to set min year for max year to work
years_bf_trl=1
abstract_list=[]
abs_count=[]
batch_size = 10
max_dwnload = 1000
    
for i in range(len(df_drug)): # for every unique drug name
    keyword = df_drug['Drug Name'][i]    #"Pembrolizumab"
  
        
    maxyear= int(df_drug['Date'][i][:4])-years_bf_trl
    
    result = Entrez.read(Entrez.esearch(db="pubmed", retmax=max_dwnload, term=keyword, sort='relevance', mindate=minyear, maxdate=maxyear, datetype='pdat')) 
    print("Total number of publications that contain the term {}: {}".format(
            keyword, result["Count"])
    )    
    # Fetch all ids
    abs_count.append(result["Count"])

    ids = result["IdList"]
    
    # only download abstracts if there are at least 5 available        
    if len(ids)>5:
       
        batches = [ids[x: x + 10] for x in range(0, len(ids), batch_size)]
        
        record_list = []
        for batch in tqdm(batches):
            h = Entrez.efetch(db="pubmed", id=batch, rettype="medline", retmode="text")
            records = Medline.parse(h)
            record_list.extend(list(records))
        
        if len(record_list) != 0: # if the recorsd list is not empty
        
            record_list_df = pd.DataFrame(record_list) # make a data frame 
                        
            record_list_df = record_list_df[record_list_df['AB'].notna()] # keep rows without na in abstract column        
            record_list_df['AB']=record_list_df['AB'].str.lower() # make all text lower case
            record_list_df['AB']=record_list_df['AB'].apply(remove_punctuation) # remove punctuation
            
            text_list = [row.split(' ') for row in record_list_df['AB']] # convert into list of single words
            
            if len(text_list) != 0: # if the list is not empty
                # remove drug names that appear in abstracts that are not the current drug
                for k in range(len(df_drug)):
                    drug_name = df_drug['Drug Name'][k] 
                    if drug_name != keyword:                         
                        for m in range(len(text_list)):
                            while drug_name in text_list[m]: text_list[m].remove(drug_name)
                                
                        
            abstract_list.extend(text_list) # add current list to rest of abstracts
           
# save abstracts
with open('abs_fiveTo1k_minus1yr.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(abstract_list, filehandle)
    

abs_count_df = pd.DataFrame(abs_count,columns=['Num abstracts found'])
abs_count_df['Drug Name'] = df_drug['Drug Name']
abs_count_df.to_csv('num_abs_per_drug.csv')# save data as csv
