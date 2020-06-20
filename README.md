# Using word embeddings to predict successful drugs 
 
# Aim
The goal was to predict immunotherapy drugs for cancer, with a focus on inhibitor checkpoints, that will be FDA approved in the future. This would give investors a better chance to asses the risks associated with investing in developing drugs and help them to make an informed decision on whether they should invest in a drug or not. 

# Requirements
Python 3.7 and the following packages:
*anaconda, Biophython, Gensim, sklearn, matplotlib, NLTK, wordcloud, Keras*

# Scripts
**sort_clinical_trials.py:** imports data from clinical trials, extracts the drug used, date and phase of each trial. Approved drugs are also added to the dataframe of drugs.

*Inputs: pZero_trials.csv, pOne_trials.csv, pTwo_trials.csv, pThree_trials.csv, pFour_trials.csv*

*Output: drugs.csv*

**dwnload_abs.py:** downloads abstracts from pubmed using the drug names in drugs.csv as search terms and the trial date for setting the max date of the search. Prepares the abstarcts for word2vec.

*Input: drugs.csv*

*Outputs: abs_fiveto1k_minus1year.data, num_abs_per_drug.csv*

**abs_word_freq.py:** reads in the abstracts, removes stopwords, finds frequency of words in abstracts for approved drugs and abstracts for developing drugs. Plots wordcloud of the words in abstracts for the approved and developing drugs seperately. 

*Input: abs_fiveto1k_minus1year.data*

*Output: wordcloud plots*

**abs_w2v.py:** reads in the abstracts, removes stopwords, trains word2vec model, saves model and plots t-SNE scatterplot of cosine similarity between an approved drug and all other drugs. 

*Input: abs_fiveto1k_minus1year.data, drugs.csv*

*Output: abs_fiveto1k_minus1year.model, plot of cosine similarity between an approved drug and all other drugs*

**pca_embeds.py:** reads in word2vec model, extracts embeddings for every word and reduces the embeddings from 100 dimensions to 2 dimensions using PCA. All drugs (approved and developing) are plotted in 2d space.

*Inputs: abs_fiveto1k_minus1year.data, drugs.csv*

*Output: PCA plot of all drugs*

**decision_classifier.py:** reads in word2vec model, extracts embeddings for every drug. Embeddings for approved drugs are given label 1, and embeddings for developing drugs are given label 0. classification of drugs is implemented using 3 different classification algorithms: Logistic Regression, Gaussian Naive Bayes and a 2-layer Sequential Neural Network from Keras. 

*Inputs: abs_fiveto1k_minus1year.data, drugs.csv*

*Output: ROC curve and confusion matrices for each classifier*





