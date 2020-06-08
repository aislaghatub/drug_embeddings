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

# Change this email to your email address
#Entrez.email = "aosulli4@ur.rochester.edu"

df_drug = pd.read_csv('drugs.csv')
df_drug = df_drug.reset_index()

years_bf_trl=2
abstract_list=[]

for i in range(len(df_drug)): # for every unique drug name
    keyword = df_drug['Drug Name'][i]    #"Pembrolizumab"
    
    maxdate= int(df_drug['Date'][i][:4])-years_bf_trl
    result = Entrez.read(Entrez.esearch(db="pubmed", retmax=10, term=keyword,sort='relevance',datetype='EDAT', mindate=1950, maxdate=maxdate)) #, datetype=(mindate, maxdate)
    print(
        "Total number of publications that contain the term {}: {}".format(
            keyword, result["Count"]
        )
    )
    
    # Fetch all ids
    MAX_COUNT = result["Count"]
    result = Entrez.read(
        Entrez.esearch(db="pubmed", retmax=result["Count"], term=keyword,sort='relevance',datetype='EDAT', mindate=1950, maxdate=maxdate)
    )
    
    ids = result["IdList"]
    batch_size = 10
    
    # keep track of drugs with zero or less than 10 abstracts
    zero_abs=[]
    less_than_ten_abs=[]
    
    if len(ids)==0:
        zero_abs.append(keyword)     
    
    elif len(ids)>0 and len(ids)<10:
        less_than_ten_abs.append(keyword)
        
    elif len(ids)>100:
        ids = ids[:100] 
       
    batches = [ids[x: x + 10] for x in range(0, len(ids), batch_size)]
    
    record_list = []
    for batch in tqdm(batches):
        h = Entrez.efetch(db="pubmed", id=batch, rettype="medline", retmode="text")
        records = Medline.parse(h)
        record_list.extend(list(records))
    print("Complete.")
    
    abstract_list.extend(record_list) # store abstracts for all drugs

# Convert to Dataframe and save
abstracts_df=pd.DataFrame(abstract_list) # convert to pandas dataframe
abstracts_df=abstracts_df[['AB','LR']] # get only abstract text and date
abstracts_df['LR']=pd.to_datetime(abstracts_df['LR']) # convert string date into datetime format
abstracts_df.columns = ['text', 'date']
abstracts_df.to_pickle('abstracts.pkl')# save data as pickle
abstracts_df.to_csv('abstracts.csv')# save data as csv

no_abs_drugs_df = pd.DataFrame(zero_abs)
no_abs_drugs_df.to_csv('drugs_with_no_abs.csv')# save data as csv

ten_abs_drugs_df = pd.DataFrame(less_than_ten_abs)
ten_abs_drugs_df.to_csv('drugs_with_less_than_ten_abs.csv')# save data as csv
