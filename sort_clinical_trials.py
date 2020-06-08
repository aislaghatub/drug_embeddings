# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:07:13 2020

@author: aosulli4
"""

# import clinical trials 

# search term: cancer
# other search term: inhibitor checkpoints
# Filters:
    # Not yet recruiting
    # Recruiting
    # Enrolled
    # Active
    # Completed
 # Studies were downloaded on 04/06/2020
   
# Not applicable phases were not included as this is defined as behvioral
# or device intervention

import pandas as pd
from datetime import datetime
import numpy as np

# make function to process trials
def get_drugs_from_trials(df, phase): 
    
    # make new df: First Posted, Interventions, Phase
    
    # find trials that use drug interventions
    df["Uses Drug"]= df["Interventions"].str.find('Drug')
    
    # delete rows (i.e., trials) that don't use drugs    
    if -1 in df.values:        
        df = df.set_index("Uses Drug")
        df = df.drop(-1, axis=0) # Delete all rows in with label -1 in 'Uses Drug' column
        df = df.reset_index()
        
    # get names of drugs used
    treatments_list = df['Interventions'].str.split('|')
    substring='Drug:'
    drug_name=[]
    for list_idx in range(len(treatments_list)):
        curr_trl = treatments_list[list_idx]
        res = list(filter(lambda x: substring in x, curr_trl))
        res[0] = res[0].lower()        # convert drug name to lower case   
        drug_name.append(res[0][6:])
        

    # each of these drugs are in phase=phase
    phaseNum = [phase] * len(drug_name)
    # get date of first posted
    new_df = pd.DataFrame(drug_name, columns=['Drug Names']) 
    new_df['Date'] = df['First Posted']
    new_df['Stage'] = phaseNum
    
    # next step: keep latest unique drug
    # sort columns by date
    new_df['Date'] =pd.to_datetime(new_df.Date)
    new_df.sort_values(by=['Date'], inplace=True) 
    return new_df

    
# Early Phase 1 / Phase 0
# 7 studies found
df_full=pd.DataFrame([])
phaseStr = ['Zero','One','Two','Three','Four']
# for each phase, get the new data frame with drugs used, date and phase
for phase in range(0,len(phaseStr)):
    
    csv_file = pd.read_csv('csv files/p' + phaseStr[phase] + '_trials.csv')
    df_phase = get_drugs_from_trials(csv_file,phase)
    df_full = df_full.append(df_phase) 
    
df_full = df_full.reset_index()    

# add approved drugs with their approval date
approved_drugs = ['ipilimumab', 'nivolumab','pembrolizumab','atezolizumab','avelumab','durvalumab','cemiplimab']
approved_str_dates =['2011-01-01','2014-01-01','2014-01-01','2016-01-01','2017-01-01','2017-01-01','2018-01-01']
approved_dates = [datetime.strptime(x, '%Y-%m-%d') for x in approved_str_dates] 
appr_drugs_data = {'Drug Names': approved_drugs, 'Date': approved_dates}
df_approved = pd.DataFrame(appr_drugs_data, columns=['Drug Names','Date'])
df_full = df_full.append(df_approved)

# sort df_full by date (up to now it was sorted by date within stage)
df_full.sort_values(by=['Date'],inplace=True)

# find unique drugs
unq_drugs=[]
unq_drugs = df_full['Drug Names'].unique()
# get the earliest date that the drugs were added to a trial
first_date=[]
stage=[]
for i in range(len(unq_drugs)):
    date_unq_drugs = df_full[df_full['Drug Names'] == unq_drugs[i]]
    #date_unq_drugs = df_full[df_full['Drug Names'].isin([unq_drugs[i]])]
    date_unq_drugs = date_unq_drugs.reset_index()
    first_date.append(date_unq_drugs['Date'][0]) # the first index is the earliest date
    stage.append(date_unq_drugs['Stage'][0])

df_drug = pd.DataFrame([])
df_drug['Drug Name'] = unq_drugs
df_drug['Date'] = first_date
df_drug['Stage'] = stage
df_drug.to_csv('drugs.csv')




