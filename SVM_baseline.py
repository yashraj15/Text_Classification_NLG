# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 19:15:38 2021

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

Corpus = pd.read_csv(r"D:\\Sem 4\\ADM\\BBC News Train.csv")

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['Text'],Corpus['Category'],test_size=0.3)

    
train_x_200_baseline = Train_X[:200]
train_y_200_baseline = Train_Y[:200]

train_x_400_baseline = Train_X[:400]
train_y_400_baseline = Train_Y[:400]

train_x_600_baseline = Train_X[:600]
train_y_600_baseline = Train_Y[:600]

train_x_800_baseline = Train_X[:800]
train_y_800_baseline = Train_Y[:800]




"""                         x            """
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)



Tfidf_vect = TfidfVectorizer(max_features=500)
Tfidf_vect.fit(Corpus['Text'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score 100%- ",accuracy_score(predictions_SVM, Test_Y)*100)
print("F1 Score(micro) 100%- ", f1_score(predictions_SVM, Test_Y, average = 'micro')*100)
print("F1 Score(macro) 100%- ", f1_score(predictions_SVM, Test_Y, average = 'macro')*100)
print("F1 Score(weighted) 100%- ", f1_score(predictions_SVM, Test_Y, average = 'weighted')*100)
print(classification_report(predictions_SVM,Test_Y))
print("------------------------------------------------------------------")

classification_report(predictions_SVM,Test_Y)

"""                         Here we use approximately 20% of the training set            """
Encoder = LabelEncoder()
train_y_200_baseline = Encoder.fit_transform(train_y_200_baseline)
Test_Y = Encoder.fit_transform(Test_Y)



Tfidf_vect = TfidfVectorizer(max_features= 200)
Tfidf_vect.fit(Corpus['Text'])
Train_X_Tfidf = Tfidf_vect.transform(train_x_200_baseline)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,train_y_200_baseline)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score 20%- ",accuracy_score(predictions_SVM, Test_Y)*100)
print("F1 Score(micro) 20%- ", f1_score(predictions_SVM, Test_Y, average = 'micro')*100)
print("F1 Score(macro) 20%- ", f1_score(predictions_SVM, Test_Y, average = 'macro')*100)
print("F1 Score(weighted) 20%- ", f1_score(predictions_SVM, Test_Y, average = 'weighted')*100)
print(classification_report(predictions_SVM,Test_Y))

print("------------------------------------------------------------------")

"""                         Here we use approximately 40% of the training set            """
Encoder = LabelEncoder()
train_y_400_baseline = Encoder.fit_transform(train_y_400_baseline)
Test_Y = Encoder.fit_transform(Test_Y)



Tfidf_vect = TfidfVectorizer(max_features= 200)
Tfidf_vect.fit(Corpus['Text'])
Train_X_Tfidf = Tfidf_vect.transform(train_x_400_baseline)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,train_y_400_baseline)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score 40%- ",accuracy_score(predictions_SVM, Test_Y)*100)
print("F1 Score(micro) 40%- ", f1_score(predictions_SVM, Test_Y, average = 'micro')*100)
print("F1 Score(macro) 40%- ", f1_score(predictions_SVM, Test_Y, average = 'macro')*100)
print("F1 Score(weighted) 40%- ", f1_score(predictions_SVM, Test_Y, average = 'weighted')*100)
classification_report(predictions_SVM,Test_Y)
print(classification_report(predictions_SVM,Test_Y))
print("------------------------------------------------------------------")


"""                         Here we use approximately 60% of the training set            """
Encoder = LabelEncoder()
train_y_600_baseline = Encoder.fit_transform(train_y_600_baseline)
Test_Y = Encoder.fit_transform(Test_Y)



Tfidf_vect = TfidfVectorizer(max_features= 200)
Tfidf_vect.fit(Corpus['Text'])  
Train_X_Tfidf = Tfidf_vect.transform(train_x_600_baseline)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,train_y_600_baseline)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score 60%- ",accuracy_score(predictions_SVM, Test_Y)*100)
print("F1 Score(micro) 60%- ", f1_score(predictions_SVM, Test_Y, average = 'micro')*100)
print("F1 Score(macro) 60%- ", f1_score(predictions_SVM, Test_Y, average = 'macro')*100)
print("F1 Score(weighted) 60%- ", f1_score(predictions_SVM, Test_Y, average = 'weighted')*100)
print(classification_report(predictions_SVM, Test_Y))
print(classification_report(predictions_SVM,Test_Y))
print("------------------------------------------------------------------")

"""                         Here we use approximately 80% of the training set            """
Encoder = LabelEncoder()
train_y_800_baseline = Encoder.fit_transform(train_y_800_baseline)
Test_Y = Encoder.fit_transform(Test_Y)



Tfidf_vect = TfidfVectorizer(max_features= 300)
Tfidf_vect.fit(Corpus['Text'])
Train_X_Tfidf = Tfidf_vect.transform(train_x_800_baseline)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,train_y_800_baseline)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score 80%- ",accuracy_score(predictions_SVM, Test_Y)*100)
print("F1 Score(micro) 80%- ", f1_score(predictions_SVM, Test_Y, average = 'micro')*100)
print("F1 Score(macro)80% - ", f1_score(predictions_SVM, Test_Y, average = 'macro')*100)
print("F1 Score(weighted) 80%- ", f1_score(predictions_SVM, Test_Y, average = 'weighted')*100)

print(classification_report(predictions_SVM,Test_Y))
classification_report(predictions_SVM,Test_Y)

