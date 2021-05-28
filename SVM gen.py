# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:09:40 2021

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


Corpus = pd.read_csv(r"D:\\Sem 4\\ADM\\BBC News Train.csv")

train_x_gen, test_x_gen, train_y_gen, test_y_gen = model_selection.train_test_split(Corpus['Text'],Corpus['Category'],test_size=0.3)

generator = pipeline('text-generation', model='gpt2')

for i, j in train_x_gen.items():
    x = generator(j, max_length = 1000, num_return_sequences=1)
    j = x[0]['generated_text']


train_x_200 = train_x_gen[:200]
train_y_200 = train_y_gen[:200]

train_x_400 = train_x_gen[:400]
train_y_400 = train_y_gen[:400]


train_x_600 = train_x_gen[:600]
train_y_600 = train_y_gen[:600]

train_x_800 = train_x_gen[:800]
train_y_800 = train_y_gen[:800]



"""                         Here we use 100% of the training set            """
Encoder = LabelEncoder()
train_y_gen = Encoder.fit_transform(train_y_gen)
test_y_gen = Encoder.fit_transform(test_y_gen)



Tfidf_vect = TfidfVectorizer(max_features=400)
Tfidf_vect.fit(Corpus['Text'])
Train_X_Tfidf = Tfidf_vect.transform(train_x_gen)
Test_X_Tfidf = Tfidf_vect.transform(test_x_gen)

#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,train_y_gen)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score 100%- ",accuracy_score(predictions_SVM, test_y_gen)*100)
print("F1 Score(micro) 100%- ", f1_score(predictions_SVM, test_y_gen, average = 'micro')*100)
print("F1 Score(macro) 100%- ", f1_score(predictions_SVM, test_y_gen, average = 'macro')*100)
print("F1 Score(weighted) 100%- ", f1_score(predictions_SVM, test_y_gen, average = 'weighted')*100)
print("------------------------------------------------------------------")

"""                         Here we use approximately 20% of the training set            """
Encoder = LabelEncoder()
train_y_200 = Encoder.fit_transform(train_y_200)
test_y_gen = Encoder.fit_transform(test_y_gen)



Tfidf_vect = TfidfVectorizer(max_features= 200)
Tfidf_vect.fit(Corpus['Text'])
Train_X_Tfidf = Tfidf_vect.transform(train_x_200)
Test_X_Tfidf = Tfidf_vect.transform(test_x_gen)

#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,train_y_200)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score 20%- ",accuracy_score(predictions_SVM, test_y_gen)*100)
print("F1 Score(micro) 20%- ", f1_score(predictions_SVM, test_y_gen, average = 'micro')*100)
print("F1 Score(macro) 20%- ", f1_score(predictions_SVM, test_y_gen, average = 'macro')*100)
print("F1 Score(weighted) 20%- ", f1_score(predictions_SVM, test_y_gen, average = 'weighted')*100)
print("------------------------------------------------------------------")

"""                         Here we use approximately 40% of the training set            """
Encoder = LabelEncoder()
train_y_400 = Encoder.fit_transform(train_y_400)
test_y_gen = Encoder.fit_transform(test_y_gen)



Tfidf_vect = TfidfVectorizer(max_features= 200)
Tfidf_vect.fit(Corpus['Text'])
Train_X_Tfidf = Tfidf_vect.transform(train_x_400)
Test_X_Tfidf = Tfidf_vect.transform(test_x_gen)

#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,train_y_400)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score 40%- ",accuracy_score(predictions_SVM, test_y_gen)*100)
print("F1 Score(micro) 40%- ", f1_score(predictions_SVM, test_y_gen, average = 'micro')*100)
print("F1 Score(macro) 40%- ", f1_score(predictions_SVM, test_y_gen, average = 'macro')*100)
print("F1 Score(weighted) 40%- ", f1_score(predictions_SVM, test_y_gen, average = 'weighted')*100)
print("------------------------------------------------------------------")


"""                         Here we use approximately 60% of the training set            """
Encoder = LabelEncoder()
train_y_600 = Encoder.fit_transform(train_y_600)
test_y_gen = Encoder.fit_transform(test_y_gen)



Tfidf_vect = TfidfVectorizer(max_features= 200)
Tfidf_vect.fit(Corpus['Text'])
Train_X_Tfidf = Tfidf_vect.transform(train_x_600)
Test_X_Tfidf = Tfidf_vect.transform(test_x_gen)

#(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,train_y_600)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score 60%- ",accuracy_score(predictions_SVM, test_y_gen)*100)
print("F1 Score(micro) 60%- ", f1_score(predictions_SVM, test_y_gen, average = 'micro')*100)
print("F1 Score(macro) 60%- ", f1_score(predictions_SVM, test_y_gen, average = 'macro')*100)
print("F1 Score(weighted) 60%- ", f1_score(predictions_SVM, test_y_gen, average = 'weighted')*100)
print("------------------------------------------------------------------")

"""                         Here we use approximately 80% of the training set            """
Encoder = LabelEncoder()
train_y_800 = Encoder.fit_transform(train_y_800)
test_y_gen = Encoder.fit_transform(test_y_gen)



Tfidf_vect = TfidfVectorizer(max_features= 200)
Tfidf_vect.fit(Corpus['Text'])
Train_X_Tfidf = Tfidf_vect.transform(train_x_800)
Test_X_Tfidf = Tfidf_vect.transform(test_x_gen)

#(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,train_y_800)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score 80% - ",accuracy_score(predictions_SVM, test_y_gen)*100)
print("F1 Score(micro) 80%- ", f1_score(predictions_SVM, test_y_gen, average = 'micro')*100)
print("F1 Score(macro) 80%- ", f1_score(predictions_SVM, test_y_gen, average = 'macro')*100)
print("F1 Score(weighted) 80%- ", f1_score(predictions_SVM, test_y_gen, average = 'weighted')*100)

print("------------------------------------------------------------------")




