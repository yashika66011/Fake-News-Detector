# -*- coding: utf-8 -*-
"""
Created on Mon May 31 20:15:55 2021

@author: pande
"""

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


#Read the data
df=pd.read_csv('C:\\news\\news.csv')


#Get shape and head
df.shape
df.head()


#Get the labels
labels=df.label
labels.head()

#Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.1, random_state=9)


# Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.2)

#DataFlair - Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)


#Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)


# Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')



# Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
print(confusion_matrix(y_test,y_pred, labels=['FAKE','REAL']))