# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 12:46:16 2021

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


#trial = Train_X.head()


count = 0
arr_text = []
arr_ind_y = []
arr_ind = []
dict1 = {}

generator('what is dead may never die', max_length=100, num_return_sequences=1)


for index, text in Train_X.items():
    
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

""" *****CONVERTING SERIES AND DF TO CSVS******************************************
trainx_series.to_csv('D:\\Sem 4\\ADM\\ADM Project\\trainx_gen.csv')
trainy_series.to_csv('D:\\Sem 4\\ADM\\ADM Project\\trainx_gen.csv')

Train_X.to_csv('D:\\Sem 4\\ADM\\ADM Project\\trainx.csv', index = False)
Train_Y.to_csv('D:\\Sem 4\\ADM\\ADM Project\\trainy.csv', index = False)

Test_X_df = Test_X.to_frame()
Test_Y_df = pd.DataFrame(Test_Y)

Test_X_df.to_csv('D:\\Sem 4\\ADM\\ADM Project\\testx.csv', index = False)
Test_Y_df.to_csv('D:\\Sem 4\\ADM\\ADM Project\\testy.csv', index = False)

trainy_df.to_csv('D:\\Sem 4\\ADM\\ADM Project\\trainy_gen.csv', index = False)
trainx_df.to_csv('D:\\Sem 4\\ADM\\ADM Project\\trainx_gen.csv', index = False)
"""




"""  ************************  200 + 200 GENERATED  ************************"""

trainx_200 = Train_X[:200]
trainx_200 = trainx_200.append(trainx_series[:200])

trainy_200 = Train_Y[:200]
trainy_200 = trainy_200.append(trainy_series[:200])

Encoder = LabelEncoder()
trainy_200 = Encoder.fit_transform(trainy_200)
Test_Y = Encoder.fit_transform(Test_Y)



Tfidf_vect = TfidfVectorizer(max_features= 200)
Tfidf_vect.fit(Corpus['Text'])
Train_X_Tfidf = Tfidf_vect.transform(trainx_200)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,trainy_200)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy

print("This is for 200 original training data + 200 generated data")
print("SVM Accuracy Score - ",accuracy_score(predictions_SVM, Test_Y)*100)
print("F1 Score(micro) - ", f1_score(predictions_SVM, Test_Y, average = 'micro')*100)
print("F1 Score(macro) - ", f1_score(predictions_SVM, Test_Y, average = 'macro')*100)

print("F1 Score(weighted) - ", f1_score(predictions_SVM, Test_Y, average = 'weighted')*100)

print("The confusion matrix is given below \n",classification_report(predictions_SVM,Test_Y))
print("------------------------------------------------------------------")


"""  ************************  400 + 400 GENERATED  ************************"""


trainx_400 = Train_X[:400]
trainx_400 = trainx_400.append(trainx_series[:400])

trainy_400 = Train_Y[:400]
trainy_400 = trainy_400.append(trainy_series[:400])

Encoder = LabelEncoder()
trainy_400 = Encoder.fit_transform(trainy_400)
Test_Y = Encoder.fit_transform(Test_Y)



Tfidf_vect = TfidfVectorizer(max_features= 200)
Tfidf_vect.fit(Corpus['Text'])
Train_X_Tfidf = Tfidf_vect.transform(trainx_400)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,trainy_400)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy

print("This is for 400 original training data + 400 generated data")
print("SVM Accuracy Score - ",accuracy_score(predictions_SVM, Test_Y)*100)
print("F1 Score(micro) - ", f1_score(predictions_SVM, Test_Y, average = 'micro')*100)
print("F1 Score(macro) - ", f1_score(predictions_SVM, Test_Y, average = 'macro')*100)
print("F1 Score(weighted) - ", f1_score(predictions_SVM, Test_Y, average = 'weighted')*100)

print("The confusion matrix is given below \n",classification_report(predictions_SVM,Test_Y))
print("------------------------------------------------------------------")




"""  ************************  800 + 800 GENERATED  ************************"""


trainx_800 = Train_X[:800]
trainx_800 = trainx_800.append(trainx_series[:800])

trainy_800 = Train_Y[:800]
trainy_800 = trainy_800.append(trainy_series[:800])

Encoder = LabelEncoder()
trainy_800 = Encoder.fit_transform(trainy_800)
Test_Y = Encoder.fit_transform(Test_Y)



Tfidf_vect = TfidfVectorizer(max_features= 200)
Tfidf_vect.fit(Corpus['Text'])
Train_X_Tfidf = Tfidf_vect.transform(trainx_800)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,trainy_800)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy

print("This is for 800 original training data + 800 generated data")
print("SVM Accuracy Score - ",accuracy_score(predictions_SVM, Test_Y)*100)
print("F1 Score(micro) - ", f1_score(predictions_SVM, Test_Y, average = 'micro')*100)
print("F1 Score(macro) - ", f1_score(predictions_SVM, Test_Y, average = 'macro')*100)
print("F1 Score(weighted) - ", f1_score(predictions_SVM, Test_Y, average = 'weighted')*100)

print("The confusion matrix is given below \n",classification_report(predictions_SVM,Test_Y))
print("------------------------------------------------------------------")




"""  ************************  1000 + 1000 GENERATED  ************************"""


trainx_1000 = Train_X
trainx_1000 = trainx_1000.append(trainx_series)

trainy_1000 = Train_Y
trainy_1000 = trainy_1000.append(trainy_series)

Encoder = LabelEncoder()
trainy_1000 = Encoder.fit_transform(trainy_1000)
Test_Y = Encoder.fit_transform(Test_Y)



Tfidf_vect = TfidfVectorizer(max_features= 500)
Tfidf_vect.fit(Corpus['Text'])
Train_X_Tfidf = Tfidf_vect.transform(trainx_1000)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,trainy_1000)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy

print("This is for 1000 original training data + 1000 generated data")
print("SVM Accuracy Score - ",accuracy_score(predictions_SVM, Test_Y)*100)
print("F1 Score(micro) - ", f1_score(predictions_SVM, Test_Y, average = 'micro')*100)
print("F1 Score(macro) - ", f1_score(predictions_SVM, Test_Y, average = 'macro')*100)
print("F1 Score(weighted) - ", f1_score(predictions_SVM, Test_Y, average = 'weighted')*100)

print("The confusion matrix is given below \n",classification_report(predictions_SVM,Test_Y))
print("------------------------------------------------------------------")



""" WHEN WE USE PURE GENERATED DATA WITH NO ORIGINAL DATA """


print("Just generated data as training data ")
Encoder = LabelEncoder()
trainy_series = Encoder.fit_transform(trainy_series)
Test_Y = Encoder.fit_transform(Test_Y)



Tfidf_vect = TfidfVectorizer(max_features= 200)
Tfidf_vect.fit(Corpus['Text'])
Train_X_Tfidf = Tfidf_vect.transform(trainx_series)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,trainy_series)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score- ",accuracy_score(predictions_SVM, Test_Y)*100)
print("F1 Score(micro) - ", f1_score(predictions_SVM, Test_Y, average = 'micro')*100)
print("F1 Score(macro) - ", f1_score(predictions_SVM, Test_Y, average = 'macro')*100)
print("F1 Score(weighted) - ", f1_score(predictions_SVM, Test_Y, average = 'weighted')*100)

print("The confusion matrix is given below \n",classification_report(predictions_SVM,Test_Y))
print("------------------------------------------------------------------")


import pygal
from IPython.display import display, HTML
from pygal.style import RedBlueStyle
from sklearn.metrics import confusion_matrix
base_html = """
<!DOCTYPE html>
<html>
  <head>
  <script type="text/javascript" src="http://kozea.github.com/pygal.js/javascripts/svg.jquery.js"></script>
  <script type="text/javascript" src="https://kozea.github.io/pygal.js/2.0.x/pygal-tooltips.min.js""></script>
  </head>
  <body>
    <figure>
      {rendered_chart}
    </figure>
  </body>
</html>
"""

def galplot(chart):
    rendered_chart = chart.render(is_unicode=True)
    plot_html = base_html.format(rendered_chart=rendered_chart)
    display(HTML(plot_html))
def plot_cm(y_true, y_pred):
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    dot_chart = pygal.Dot(x_label_rotation=30, interpolate='cubic', style=RedBlueStyle)
    dot_chart.title = 'Confusion Matrix'
    dot_chart.x_labels = labels
    dot_chart.x_title = "Predicted"
    dot_chart.y_title = "Actual"
    for i in range(len(labels)):
        dot_chart.add(labels[i], cm[i,:])
    galplot(dot_chart)
    
plot_cm(Test_Y, predictions_SVM)



