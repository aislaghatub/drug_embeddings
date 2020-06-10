# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:46:37 2020

@author: aosulli4
"""

# using biophython
from Bio import Entrez
from Bio import Medline
from tqdm import tqdm
import pandas as pd
import string

# Change this email to your email address
#Entrez.email = "aosulli4@ur.rochester.edu"

# remove punctuation
def remove_punctuation(s):
    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
    return s

df_drug = pd.read_csv('drugs.csv')
df_drug = df_drug.reset_index()

abstracts_df=pd.DataFrame(columns=['text']) # make pandas dataframe
clean_abstracts_df=pd.DataFrame(columns=['text']) # make pandas dataframe

# find year of oldest drug in list
# they are sorted by date so it corresponds to the first index
oldest_drug_date = df_drug['Date'][0][:4]

years_bf_trl=1
abstract_list=[]
abs_count=[]
zero_abs=[]
less_than_ten_abs=[]
    
for i in range(len(df_drug)): # for every unique drug name
    keyword = df_drug['Drug Name'][i]    #"Pembrolizumab"
    
    # maxdate= oldest_drug_date
    maxdate= int(df_drug['Date'][i][:4])-years_bf_trl
    result = Entrez.read(Entrez.esearch(db="pubmed", retmax=10, term=keyword,sort='relevance',datetype='EDAT', maxdate=maxdate)) #, datetype=(mindate, maxdate)
    print(
        "Total number of publications that contain the term {}: {}".format(
            keyword, result["Count"]
        )
    )
    
    # Fetch all ids
    abs_count.append(result["Count"])
    result = Entrez.read(
        Entrez.esearch(db="pubmed", retmax=result["Count"], term=keyword,sort='relevance',datetype='EDAT', maxdate=maxdate)
    )
    
    ids = result["IdList"]
    batch_size = 10
    
    # keep track of drugs with zero or less than 10 abstracts  
    if len(ids)==0:
        zero_abs.append(keyword)     
    
    elif len(ids)>0 and len(ids)<10:
        less_than_ten_abs.append(keyword)
        
    elif len(ids)>500:
        ids = ids[:500] 
       
    batches = [ids[x: x + 10] for x in range(0, len(ids), batch_size)]
    
    record_list = []
    for batch in tqdm(batches):
        h = Entrez.efetch(db="pubmed", id=batch, rettype="medline", retmode="text")
        records = Medline.parse(h)
        record_list.extend(list(records))
    print("Complete.")
    
    if len(record_list) != 0: # if the recorsd list is not empty
        record_list_df = pd.DataFrame(record_list)
           
        record_list_df=record_list_df.dropna() # drop rows that have nan value 
        # record_list_df=record_list_df[~record_list_df.text.str.contains("nan")] #or have 'nan' string in them
                
        record_list_df['AB']=record_list_df['AB'].str.lower() # make all text lower case
        record_list_df['AB']=record_list_df['AB'].apply(remove_punctuation) # remove punctuation
        
        text_list = [row.split(' ') for row in record_list_df['AB']] # convert into list of single words
        
        if len(text_list) != 0:
            # remove drug names that appear in abstracts that are not the current drug
            for k in range(len(df_drug)):
                drug_name = df_drug['Drug Name'][k] 
                if drug_name != keyword and drug_name in text_list:
                    text_list.remove(drug_name)
                    
            text_df = pd.DataFrame(text_list)       
            clean_abstracts_df['text'].append(text_df)
           
# save

clean_abstracts_df.to_pickle('clean_abstracts_500.pkl')# save data as pickle
clean_abstracts_df.to_csv('clean_abstracts_500.csv')# save data as csv

no_abs_drugs_df = pd.DataFrame(zero_abs)
no_abs_drugs_df.to_csv('drugs_with_no_abs.csv')# save data as csv

ten_abs_drugs_df = pd.DataFrame(less_than_ten_abs)
ten_abs_drugs_df.to_csv('drugs_with_less_than_ten_abs.csv')# save data as csv

abs_count_df = pd.DataFrame(abs_count,columns=['Num abstracts found'])
abs_count_df['Drug Name'] = df_drug['Drug Name']
abs_count_df.to_csv('num_abs_per_drug.csv')# save data as csv
