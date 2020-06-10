# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:57:40 2020

@author: aosulli4
"""

# given word embeddings calculate probability of FDA approval
# supervised learning, labelled data as 1 approved, 0 not approved

import pandas as pd
import numpy as np
from gensim.models import Word2Vec

# for classification
from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV
from sklearn.metrics import auc,roc_auc_score, plot_roc_curve,precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# load w2v model
model = Word2Vec.load("w2v_100abs_model_1year.model")

# load known drug names
df_drug = pd.read_csv('drugs.csv')
other_drugs=df_drug['Drug Name'].tolist() # convert series to list
drug_stages=df_drug['Stage'].tolist() # convert series to list

approved_drugs = ['ipilimumab', 'nivolumab','pembrolizumab','atezolizumab','avelumab','durvalumab','cemiplimab']

# get drug emabeddings
app_embed=[]
for i in range(len(approved_drugs)): # no data on last drug
    if approved_drugs[i] in model.wv.vocab:
        app_embed.append(model.wv.word_vec(approved_drugs[i]))
    
other_embed=[]
for i in range(len(other_drugs)):
    if other_drugs[i] in model.wv.vocab: # if drug name is in the model
       other_embed.append(model.wv.word_vec(other_drugs[i])) # get its vetor, normalised

all_embeds = app_embed + other_embed       
x = pd.DataFrame(all_embeds)   

# label approved + phase3 & 4 drugs as approved to help balance classes
# late_stage_drug_idx = [i for i, x in enumerate(drug_stages) if (x == 3.0 or x == 4.0)]
# late_stage_drug = [other_drugs[i] for i in late_stage_drug_idx]

    
list_y = [1]*len(all_embeds) # all drugs are given label 0
one_labels = 0 

list_y[:len(approved_drugs)-1]= [one_labels for i in range(len(approved_drugs)-1)]  # approved drugs are given label 1

# # late stage drugs are given label 1
# for i in range(len(late_stage_drug_idx)):
#     list_y[late_stage_drug_idx[i]] = one_labels 

y = pd.DataFrame(list_y)

classifier=LogisticRegression(solver='lbfgs',class_weight='balanced')

# Outer cross-validation
cv = StratifiedKFold(n_splits=2)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
# Outer crossvalidation loop
fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(x, y)):         
    classifier.fit(x.iloc[train], y.iloc[train])  # train model 
    
    viz = plot_roc_curve(classifier, x.iloc[test], y.iloc[test],
                         name='ROC fold {}'.format(i),
                          alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)                    
    
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating characteristic example")
ax.legend(loc="lower right")
plt.show()


# cmd = ConfusionMatrixDisplay(confusion_matrix=kfs.mean_conf_mat,
#                               display_labels=classifier.classes_)

# conf_mat = confusion_matrix(y_true, y_pred,normalize='true')
# cmd.plot(include_values=True, xticks_rotation='horizontal', values_format=None,cmap='viridis', ax=ax)
# cmd.ax_.set_title('Confusion Matrix')