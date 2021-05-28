# -*- coding: utf-8 -*-
"""
Created on Tue May  4 09:56:24 2021

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
from sklearn.metrics import classification_report
from transformers import pipeline, set_seed

Corpus = pd.read_csv(r"D:\\Sem 4\\ADM\\BBC News Train.csv")

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['Text'],
                                                                    Corpus['Category'],
                                                                    test_size=0.3)

trainx40 = Train_X[:40]
trainy40 = Train_Y[:40]

"""         40          """

Encoder = LabelEncoder()
trainy40 = Encoder.fit_transform(trainy40)
Test_Y = Encoder.fit_transform(Test_Y)



Tfidf_vect = TfidfVectorizer(max_features= 200)
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
print("SVM Accuracy Score  ",accuracy_score(predictions_SVM, Test_Y)*100)
print("F1 Score(micro) -", f1_score(predictions_SVM, Test_Y, average = 'micro')*100)
print("F1 Score(macro) - ", f1_score(predictions_SVM, Test_Y, average = 'macro')*100)
print("F1 Score(weighted) - ", f1_score(predictions_SVM, Test_Y, average = 'weighted')*100)
print(classification_report(predictions_SVM,Test_Y))

print("------------------------------------------------------------------")



trainx60 = Train_X[:60]
trainy60 = Train_Y[:60]

"""         60          """

Encoder = LabelEncoder()
trainy60 = Encoder.fit_transform(trainy60)
Test_Y = Encoder.fit_transform(Test_Y)



Tfidf_vect = TfidfVectorizer(max_features= 200)
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
print("SVM Accuracy Score  ",accuracy_score(predictions_SVM, Test_Y)*100)
print("F1 Score(micro) -", f1_score(predictions_SVM, Test_Y, average = 'micro')*100)
print("F1 Score(macro) - ", f1_score(predictions_SVM, Test_Y, average = 'macro')*100)
print("F1 Score(weighted) - ", f1_score(predictions_SVM, Test_Y, average = 'weighted')*100)
print(classification_report(predictions_SVM,Test_Y))

print("------------------------------------------------------------------")


trainx80 = Train_X[:80]
trainy80 = Train_Y[:80]

"""         80         """

Encoder = LabelEncoder()
trainy80 = Encoder.fit_transform(trainy80)
Test_Y = Encoder.fit_transform(Test_Y)



Tfidf_vect = TfidfVectorizer(max_features= 200)
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
print("SVM Accuracy Score  ",accuracy_score(predictions_SVM, Test_Y)*100)
print("F1 Score(micro) -", f1_score(predictions_SVM, Test_Y, average = 'micro')*100)
print("F1 Score(macro) - ", f1_score(predictions_SVM, Test_Y, average = 'macro')*100)
print("F1 Score(weighted) - ", f1_score(predictions_SVM, Test_Y, average = 'weighted')*100)
print(classification_report(predictions_SVM,Test_Y))

print("------------------------------------------------------------------")

trainx100 = Train_X[:100]
trainy100 = Train_Y[:100]

"""         100         """

Encoder = LabelEncoder()
trainy100 = Encoder.fit_transform(trainy100)
Test_Y = Encoder.fit_transform(Test_Y)



Tfidf_vect = TfidfVectorizer(max_features= 200)
Tfidf_vect.fit(Corpus['Text'])
Train_X_Tfidf = Tfidf_vect.transform(trainx100)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,trainy100)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score  ",accuracy_score(predictions_SVM, Test_Y)*100)
print("F1 Score(micro) -", f1_score(predictions_SVM, Test_Y, average = 'micro')*100)
print("F1 Score(macro) - ", f1_score(predictions_SVM, Test_Y, average = 'macro')*100)
print("F1 Score(weighted) - ", f1_score(predictions_SVM, Test_Y, average = 'weighted')*100)
print(classification_report(predictions_SVM,Test_Y))

print("------------------------------------------------------------------")