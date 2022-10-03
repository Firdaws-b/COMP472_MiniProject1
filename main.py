# This is a sample Python script.
import sys
from collections import OrderedDict

import datasets as dataSets
import numpy as np
import json
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import gensim
import nltk
from matplotlib import pyplot as plt
from collections import Counter

from matplotlib.pyplot import clf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import tree, metrics, decomposition, svm
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
#np.set_printoptions(threshold=sys.maxsize)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction import DictVectorizer
# from pandas.io import json


pathToFile = "/Users/kawtherbouzeghaya/Downloads/goemotions.json"

redditData = pd.read_json(pathToFile)
file = open(pathToFile)
loadedData = json.load(file)
sentimentClasses = ['positive', 'neutral', 'negative', 'ambiguous']
myRepeatedEmotionsList = redditData[1]  # We first need to get the list of emotions from the data set
# Then we will need to remove the repeated values and convert the list back.

myUnrepeatedEmotionsClassesList = dict.fromkeys(myRepeatedEmotionsList)

myRepeatedEmotionsListSorted = np.sort(myRepeatedEmotionsList)  # Used to sort the emotions' list by alphabetical order

myUnrepeatedEmotionsClassesList = np.unique(
    myRepeatedEmotionsListSorted).tolist()  # used to get the labels of the pie chart.

repeatedEmotions = dict(
    Counter(myRepeatedEmotionsListSorted))  # Get the number of each type of emotions for every post.

# At first I created the histogram, but it was too ugly :( so I had to switch to pie charts !
# plt.hist(myRepeatedEmotionsList,bins=28)
# plt.show()


# print(myUnrepeatedEmotionsClassesList)
# print(redditData)
numberOfPositiveSentiments = redditData[2].value_counts()['positive']
numberOfNegativeSentiments = redditData[2].value_counts()['negative']
numberOfAmbiguousSentiments = redditData[2].value_counts()['ambiguous']
numberOfNeutralSentiments = redditData[2].value_counts()['neutral']
# This print is used
# print(repeatedEmotions)
# Labels
sentimentClassesData = [numberOfPositiveSentiments, numberOfNegativeSentiments, numberOfAmbiguousSentiments,
                        numberOfNeutralSentiments]
# Plotting the pie chart of emotions
numberOfEmotions = []
labelsOfEmotions = []
for x, y in repeatedEmotions.items():
    labelsOfEmotions.append(x)
    numberOfEmotions.append(y)

emotionsPieChart = plt.figure(figsize=(10, 7))
plt.pie(numberOfEmotions, labels=labelsOfEmotions, autopct='%1.0f%%')
plt.title('Reddit Posts Emotions Distribution.')

sentimentPieChart = plt.figure(figsize=(10, 7))  # Set the size of the figure
plt.pie(sentimentClassesData, labels=sentimentClasses, autopct='%1.0f%%')
plt.title('Reddit Posts Sentiments Distribution.')
plt.show()

# Part Two:
print(redditData)
dataSets = redditData[0]
vectorizer = CountVectorizer()
vectorizer.fit_transform(dataSets)
print("Tokens: ", vectorizer.vocabulary_)
X = vectorizer.fit_transform(dataSets)
print(vectorizer.get_feature_names_out())
print(X.toarray())


#dataArray = vectorizer.get_feature_names_out()
#print(dataArray)


print("Decision Tree analysis using tree.DecisionTreeClassifier")
# Emotions
# Data sorting
print("------------------ Emotions Decision Tree analysis------------")
Y = vectorizer.fit_transform(redditData[1])
print(vectorizer.get_feature_names_out())
print(Y)
print("---------------------------")
print(X)
enc = LabelEncoder()
Y=enc.fit_transform(redditData[1])
print(Y)
ndf = redditData
ndf[1]= np.transpose(Y)
print(ndf)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# Decision Tree
#create and print the decision tree
emotionsDecisionTree = tree.DecisionTreeClassifier()
emotionsDecisionTree.fit(X_train,Y_train)
emotionsDecisionTrePrediction = emotionsDecisionTree.predict(X_test)
print("Emotion's Accuracy(without criterion)  :", metrics.accuracy_score(Y_test, emotionsDecisionTrePrediction))


#Sentiments
print("-------- Sentiments decision Tree analysis")
# Data sorting
Z = vectorizer.fit_transform(redditData[2])
print(vectorizer.get_feature_names_out())
print("---------------------------")
print(Y)
print("---------------------------")
print(X)
enc = LabelEncoder()
Z=enc.fit_transform(redditData[2])
print(Y)
ndf = redditData
ndf[2]= np.transpose(Y)
print(ndf)
X_train, X_test, Z_train, Z_test = train_test_split(X, Z, test_size=0.2)

#Decision Tree
sentimentDecisionTree = tree.DecisionTreeClassifier()
sentimentDecisionTree.fit(X_train, Z_train)
sentimentDecisionTreePrediction = sentimentDecisionTree.predict(X_test)
print("Sentiments' Accuracy(without criterion)  :", metrics.accuracy_score(Z_test, sentimentDecisionTreePrediction))

#Top-DT
print(" Decision Tree analysis (Hyper parameters)using Grid search")

# Setting up our tuning parameters.
parametersGrid = { 'criterion':('entropy','gini'), 'max_depth':(4,6), 'min_sample_split':(2,3,6)}
grid = GridSearchCV(clf, param_grid=parametersGrid, cv=5)
#Emotion
print("------Emotions Decision Tree analysis (hyper parameters)--------")
grid.fit(X_train, Y_train)
#print(" Tuned Decision Tree parameters:".format(emotionsDecisionTree.best_params_))
#print("Best score is {}".format(emotionsDecisionTree.best_score))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
