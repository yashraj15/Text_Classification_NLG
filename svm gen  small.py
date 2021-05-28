# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:23:39 2021

@author: yashr
"""

import pandas as pd
import numpy as np
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from transformers import pipeline, set_seed
from sklearn.metrics import classification_report
#load dataset
Corpus = pd.read_csv(r"D:\\Sem 4\\ADM\\BBC News Train.csv")

#divide into test and train
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['Text'],Corpus['Category'],test_size=0.3)

#start lang gen
generator = pipeline('text-generation', model='gpt2')



count = 0
arr_text = []
arr_ind_y = []
arr_ind = []
dict1 = {}
for index, text in Train_X[:80].items():
    
    print("This is Article number: ", count)
    init_txt_len = len(text)
    print("len of base text:", init_txt_len)
    temp = generator(text, max_length=1000, num_return_sequences=1)
    gen_text = temp[0]['generated_text']
    
    print("len of gen text", len(gen_text))
    gen_text_len = len(temp[0]['generated_text'])
    new_entry = gen_text[init_txt_len:]
    
    arr_text.append(new_entry)
    arr_ind_y.append(Train_Y[index])
    arr_ind.append(index)
    count = count + 1    
    
dict_train_x = dict(zip(arr_ind, arr_text))
trainx_series = pd.Series(dict_train_x)
trainx_df = trainx_series.to_frame()
trainx_df.rename(columns = {0: 'Text'})

dict_train_y = dict(zip(arr_ind, arr_ind_y))
trainy_series = pd.Series(dict_train_y)
trainy_df = trainy_series.to_frame()
trainy_df.rename(columns = {0: 'Text'})


"""  ************************  20 + 20 GENERATED  ************************"""
trainx20 = Train_X[:20]
trainx20 = trainx20.append(trainx_series[:20])


trainy20 = Train_Y[:20]
trainy20 = trainy20.append(trainy_series[:20])


Encoder = LabelEncoder()
trainy20 = Encoder.fit_transform(trainy20)
Test_Y = Encoder.fit_transform(Test_Y)



Tfidf_vect = TfidfVectorizer(max_features= 1200)
Tfidf_vect.fit(Corpus['Text'])
Train_X_Tfidf = Tfidf_vect.transform(trainx20)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,trainy20)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy

print("This is for 20 original training data + 20 generated data")
print("SVM Accuracy Score - ",accuracy_score(predictions_SVM, Test_Y)*100)
print("F1 Score(micro) - ", f1_score(predictions_SVM, Test_Y, average = 'micro')*100)
print("F1 Score(macro) - ", f1_score(predictions_SVM, Test_Y, average = 'macro')*100)

print("F1 Score(weighted) - ", f1_score(predictions_SVM, Test_Y, average = 'weighted')*100)

print("The confusion matrix is given below \n",classification_report(predictions_SVM,Test_Y))
print("------------------------------------------------------------------")


"""  ************************  20 + 40 GENERATED  ************************"""
trainx40 = Train_X[:20]
trainx40 = trainx40.append(trainx_series[:40])


trainy40 = Train_Y[:20]
trainy40 = trainy40.append(trainy_series[:40])


Encoder = LabelEncoder()
trainy40 = Encoder.fit_transform(trainy40)
Test_Y = Encoder.fit_transform(Test_Y)



Tfidf_vect = TfidfVectorizer(max_features= 1200)
Tfidf_vect.fit(Corpus['Text'])
Train_X_Tfidf = Tfidf_vect.transform(trainx40)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,trainy40)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy

print("This is for 20 original training data + 40 generated data")
print("SVM Accuracy Score - ",accuracy_score(predictions_SVM, Test_Y)*100)
print("F1 Score(micro) - ", f1_score(predictions_SVM, Test_Y, average = 'micro')*100)
print("F1 Score(macro) - ", f1_score(predictions_SVM, Test_Y, average = 'macro')*100)

print("F1 Score(weighted) - ", f1_score(predictions_SVM, Test_Y, average = 'weighted')*100)

print("The confusion matrix is given below \n",classification_report(predictions_SVM,Test_Y))
print("------------------------------------------------------------------")



"""  ************************  20 + 60 GENERATED  ************************"""
trainx60 = Train_X[:20]
trainx60 = trainx60.append(trainx_series[:60])


trainy60 = Train_Y[:20]
trainy60 = trainy60.append(trainy_series[:60])


Encoder = LabelEncoder()
trainy60 = Encoder.fit_transform(trainy60)
Test_Y = Encoder.fit_transform(Test_Y)



Tfidf_vect = TfidfVectorizer(max_features= 900)
Tfidf_vect.fit(Corpus['Text'])
Train_X_Tfidf = Tfidf_vect.transform(trainx60)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,trainy60)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy

print("This is for 20 original training data + 60 generated data")
print("SVM Accuracy Score - ",accuracy_score(predictions_SVM, Test_Y)*100)
print("F1 Score(micro) - ", f1_score(predictions_SVM, Test_Y, average = 'micro')*100)
print("F1 Score(macro) - ", f1_score(predictions_SVM, Test_Y, average = 'macro')*100)

print("F1 Score(weighted) - ", f1_score(predictions_SVM, Test_Y, average = 'weighted')*100)

print("The confusion matrix is given below \n",classification_report(predictions_SVM,Test_Y))
print("------------------------------------------------------------------")



"""  ************************  20 + 80 GENERATED  ************************"""
trainx80 = Train_X[:20]
trainx80 = trainx80.append(trainx_series[:80])


trainy80 = Train_Y[:20]
trainy80 = trainy80.append(trainy_series[:80])


Encoder = LabelEncoder()
trainy80 = Encoder.fit_transform(trainy80)
Test_Y = Encoder.fit_transform(Test_Y)



Tfidf_vect = TfidfVectorizer(max_features= 1500)
Tfidf_vect.fit(Corpus['Text'])
Train_X_Tfidf = Tfidf_vect.transform(trainx80)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,trainy80)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy

print("This is for 20 original training data + 80 generated data")
print("SVM Accuracy Score - ",accuracy_score(predictions_SVM, Test_Y)*100)
print("F1 Score(micro) - ", f1_score(predictions_SVM, Test_Y, average = 'micro')*100)
print("F1 Score(macro) - ", f1_score(predictions_SVM, Test_Y, average = 'macro')*100)

print("F1 Score(weighted) - ", f1_score(predictions_SVM, Test_Y, average = 'weighted')*100)

print("The confusion matrix is given below \n",classification_report(predictions_SVM,Test_Y))
print("------------------------------------------------------------------")