# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:57:40 2020

@author: aosulli4
"""

# given word embeddings calculate probability of FDA approval
# supervised learning, labelled data as 1 approved, 0 not approved

# seeding the random numbers to get consistent results from neural network
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)


import pandas as pd
import numpy as np
from gensim.models import Word2Vec

# for classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc,roc_auc_score, plot_roc_curve #,precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  #,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# ----------------------Prepare Data for classification ----------------

# load w2v model
model = Word2Vec.load("abs_fiveTofifty_minus1yr.model")

# load known drug names
df_drug = pd.read_csv('drugs.csv')
other_drugs=df_drug['Drug Name'].tolist() # convert series to list
drug_stages=df_drug['Stage'].tolist() 

approved_drugs = ['ipilimumab', 'nivolumab','pembrolizumab','atezolizumab','avelumab','durvalumab','cemiplimab']

# Get drug emabeddings vectors/ features
app_embed=[]
for drug in approved_drugs: 
    if drug in model.wv.vocab: # if drug name is in the model
        app_embed.append(model.wv.word_vec(drug)) # get its vetor
        
other_embed=[]
for drug in other_drugs:
    if drug in model.wv.vocab: 
       other_embed.append(model.wv.word_vec(drug))                        
all_embeds = app_embed + other_embed       
x=np.array(all_embeds)

# Prepare classification labels
list_y=[1]*len(app_embed)+[0]*(len(all_embeds)-len(app_embed))
y = np.array(list_y)

# ----------------------Classifier fitting - Logistic Regression ----------------
# See:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
# Outer cross-validation
tprs = []
aucs = []
y_prob_lst=[];y_pred_lst=[]; y_true_lst=[];
mean_fpr = np.linspace(0, 1, 100)
classifier=LogisticRegression(solver='lbfgs',class_weight='balanced')
cv = StratifiedKFold(n_splits=4)
fig, ax = plt.subplots()

# Outer crossvalidation loop
for i, (train, test) in enumerate(cv.split(x, y)):         
    classifier.fit(x[train], y[train])  # train model 
    
    viz = plot_roc_curve(classifier, x[test], y[test],
                         name='ROC fold {}'.format(i),
                          alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc) 

    y_prob_lst.extend(classifier.predict_proba(x[test])[0]) # get probabilities;   
    y_pred_lst.extend(classifier.predict(x[test])) # get labels                   
    y_true_lst.extend(y[test])
    
# Plot AUC        
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
       title="Receiver operating characteristic")
ax.legend(loc="lower right")
plt.title('Logistic Regression')
plt.show()

log_reg_fpr = mean_fpr
log_reg_tpr = mean_tpr

# Plot Confusion Matrix
CM=confusion_matrix(y_true_lst, y_pred_lst,normalize='true') # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
CM_D=ConfusionMatrixDisplay(confusion_matrix=CM,display_labels=classifier.classes_)
fig, ax = plt.subplots()
CM_D.plot(include_values=True, xticks_rotation='horizontal', values_format=None,cmap='viridis', ax=ax)  
CM_D.ax_.set_title('LR Confusion Matrix')
    

# ----------------------Classifier fitting - Gaussian Naive Bayes Classifier ----------------
# See:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
# Outer cross-validation
tprs = []
aucs = []
y_prob_lst=[];y_pred_lst=[]; y_true_lst=[];
mean_fpr = np.linspace(0, 1, 100)
classifier=GaussianNB()
cv = StratifiedKFold(n_splits=4)
fig, ax = plt.subplots()

# Outer crossvalidation loop
for i, (train, test) in enumerate(cv.split(x, y)):         
    

    # option 2: only single level crossval
    classifier.fit(x[train], y[train])  # train model 

    viz = plot_roc_curve(classifier, x[test], y[test],
                         name='ROC fold {}'.format(i),
                          alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc) 

  #  y_prob_lst.extend(classifier.predict_proba(x[test])[0]) # get probabilities;   
    y_pred_lst.extend(classifier.predict(x[test])) # get labels                   
    y_true_lst.extend(y[test])
    
# Plot AUC        
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
       title="Receiver operating characteristic")
ax.legend(loc="lower right")
plt.title('NB Classifier')
plt.show()

nb_fpr = mean_fpr
nb_tpr = mean_tpr

# Plot Confusion Matrix
CM=confusion_matrix(y_true_lst, y_pred_lst,normalize='true') # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
CM_D=ConfusionMatrixDisplay(confusion_matrix=CM,display_labels=classifier.classes_)
fig, ax = plt.subplots()
CM_D.plot(include_values=True, xticks_rotation='horizontal', values_format=None,cmap='viridis', ax=ax)  
CM_D.ax_.set_title('MLP Confusion Matrix')


# ----------------------Sequential model With Keras ----------------
# See:
#  https://machinelearningmastery.com/cost-sensitive-neural-network-for-imbalanced-classification/
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
# Outer cross-validation

from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import roc_curve

kmodel = Sequential()
n_input = x[train].shape[1]

kmodel.add(Dense(10, input_dim=n_input, activation='relu', kernel_initializer='he_uniform')) # 10 neurons in hidden layer
kmodel.add(Dense(1, activation='sigmoid'))  # output layer
# model fit using stochastic gradient descent with the default learning rate and optimized according to cross-entropy loss
kmodel.compile(loss='binary_crossentropy', optimizer='sgd')

tprs = []
aucs = []
y_prob_lst=[];y_pred_lst=[]; y_true_lst=[];
mean_fpr = np.linspace(0, 1, 100)
cv = StratifiedKFold(n_splits=4)
fig, ax = plt.subplots()

# Outer crossvalidation loop
for i, (train, test) in enumerate(cv.split(x, y)):    
     
    # weights: instead of minimising squared error, it minimises misclassification cost    
    weight_approved = round(len(other_embed)/len(app_embed))
    weights = {0:1, 1:weight_approved}
    
    history = kmodel.fit(x[train], y[train], class_weight=weights, epochs=1000, verbose=0)

    yhat = kmodel.predict(x[test])
    aucs.append(roc_auc_score(y[test], yhat))

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y[test], yhat)
    interp_tpr = np.interp(mean_fpr, fpr_keras, tpr_keras)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc) 
    
    threshold=0.5
    y_pred_labels = np.where(yhat > threshold, 1,0)
    y_pred_lst.extend(y_pred_labels ) # get labels                   
    y_true_lst.extend(y[test])
    
# Plot AUC        
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
        title="Receiver operating characteristic")
ax.legend(loc="lower right")
plt.title('Neural Network')
plt.show()

nn_fpr = mean_fpr
nn_tpr = mean_tpr

#Plot Confusion Matrix
CM=confusion_matrix(y_true_lst, y_pred_lst,normalize='true') # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
CM_D=ConfusionMatrixDisplay(confusion_matrix=CM,display_labels=['not approved','approved']) #display_labels=classifier.classes_
fig, ax = plt.subplots()
CM_D.plot(include_values=True, xticks_rotation='horizontal', values_format=None,cmap='viridis', ax=ax) 
CM_D.ax_.set_title('NN Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')

# ---------------------- Compare 3 classifiers --------------------------------
# plot AUC comparison of all 3 classifiers
log_reg_auc = auc(log_reg_fpr, log_reg_tpr)
nb_auc = auc(nb_fpr, nb_tpr)
nn_auc = auc(nn_fpr, nn_tpr)

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

ax.plot(log_reg_fpr, log_reg_tpr, color='blue',
        label=r'LR Mean ROC (AUC = %0.2f)' % (log_reg_auc),
        lw=2, alpha=.8)
ax.plot(nb_fpr, nb_tpr, color='orange',
        label=r'NB Mean ROC (AUC = %0.2f)' % (nb_auc),
        lw=2, alpha=.8)
ax.plot(nn_fpr, nn_tpr, color='green',
        label=r'NN Mean ROC (AUC = %0.2f)' % (nn_auc),
        lw=2, alpha=.8)

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="Receiver operating characteristics year-1")
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc="lower right")
plt.show()









