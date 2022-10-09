# This is a sample Python script.
import sys
from collections import OrderedDict

import numpy as np
import json
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import gensim
import nltk
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from collections import Counter
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

performance = open('performance', 'w')
performance.write("Part 2.4 of the mini project \n")
# precision.write("Emotions: ")
# precision.write("Confusion Matrix:")
# precision.write("Multi-Layered Perceptron Classifier with default parameters, except the number of epochs:  \n")
# precision.write("Precision: \n")
# precision.write("Multi-Layered Perceptron Classifier Using GridSearchCv: \n")


pathToFile = "/Users/firdawsbouzeghaya/Downloads/goemotions (1).json"

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

numberOfPositiveSentiments = redditData[2].value_counts()['positive']
numberOfNegativeSentiments = redditData[2].value_counts()['negative']
numberOfAmbiguousSentiments = redditData[2].value_counts()['ambiguous']
numberOfNeutralSentiments = redditData[2].value_counts()['neutral']

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
# Processing the dataset (idea 1) :
#######################################################
# corpus = np.array(loadedData)  # This returns a multidimensional array.
# flatten_array = corpus.flatten()  # We need to transform our array to a one dimensional array.
# vectorizer = CountVectorizer()  # Can we use DictVectorizer instead of count ?
# X = vectorizer.fit_transform(flatten_array)
# X.toarray()
# print('The total number of tokens in the dataset is:', len(vectorizer.get_feature_names_out()))
# Processing the dataset(idea to keep )
######################################################
# print(redditData)
dataSets = redditData[0]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dataSets)
# print(vectorizer.get_feature_names_out())
print("Total Number of Tokens: ", len(vectorizer.get_feature_names_out()))
# print(X.toarray()) why is the output not showing !!
print(X[0])

# Multinomial Naive-Bayes for Emotions:
print('----------------------------------------------------')
print('Multinomial Naive-Bayes for emotions:')
Y = vectorizer.fit_transform(redditData[1])  # Y is the emotions label.
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

Y = enc.fit_transform(redditData[1])
# print(Y)
ndf = redditData
ndf[1] = np.transpose(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X_train.toarray())
print(len(X_train.toarray()))

mnb = MultinomialNB()
mnb.fit(X_train, Y_train)
y_pred_MNB_emotions = mnb.predict(X_test)
print(y_pred_MNB_emotions)
print("Accuracy of the dataset using emotions as a target using Multinomial Naive-Bayes is: ",
      metrics.accuracy_score(Y_test, y_pred_MNB_emotions))
# print("Confusion Matrix for emotions:\n", confusion_matrix(Y_test, y_pred_MNB_emotions))
performance.write("Confusion Matrix of emotions using MNB: \n")
performance.write("Confusion Matrix of MNB emotions:\n")
print(confusion_matrix(Y_test, y_pred_MNB_emotions), file=performance)
performance.write("Classification report of MNB emotions: \n")
print(classification_report(Y_test, y_pred_MNB_emotions), file=performance)

# Multinomial Naive-Bayes for sentiments:
print('----------------------------------------------------')
print('Multinomial Naive-Bayes for sentiments:')
Z = enc.fit_transform(redditData[2])
ndf = redditData
ndf[2] = np.transpose(Z)
X_train, X_test, Z_train, Z_test = train_test_split(X, Z, test_size=0.2)
mnb.fit(X_train, Z_train)
z_pred_sentiments_MNB = mnb.predict(X_test)
print(z_pred_sentiments_MNB)
print("Accuracy of the dataset using sentiments as a target using Multinomial Naive-Bayes is: ",
      metrics.accuracy_score(Z_test, z_pred_sentiments_MNB))
# print("Confusion Matrix for sentiments:\n", confusion_matrix(Z_test, z_pred_sentiments_MNB))
performance.write("Confusion Matrix of sentiments using MNB: \n")
performance.write("Confusion Matrix of MNB sentiments:\n")
print(confusion_matrix(Z_test, z_pred_sentiments_MNB), file=performance)
confusion_matrix(Y_test, y_pred_MNB_emotions)
performance.write("Classification report of MNB sentiments: \n")
print(classification_report(Z_test, z_pred_sentiments_MNB), file=performance)
print('----------------------------------------------------')
print('Multi-Layered Perceptron for emotions: ')
# Import MLPClassifer
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(max_iter=1)
clf.fit(X_train, Y_train)
y_pred_MLP_emotions = clf.predict(X_test)
print("Accuracy of the dataset using emotions as a target using Multi-Layered Perceptron is: ",
      metrics.accuracy_score(Y_test, y_pred_MLP_emotions))
# Implementation of Confusion Matrix of MLP:
# print("Confusion Matrix for emotions:\n", confusion_matrix(Y_test, y_pred_MLP_emotions))
performance.write("Confusion Matrix of MLP emotions:\n")
print(confusion_matrix(Y_test, y_pred_MLP_emotions, ), file=performance)
performance.write("Classification report of MLP emotions: \n")
print(classification_report(Y_test, y_pred_MLP_emotions), file=performance)
print('----------------------------------------------------')
print('Multi-Layered Perceptron for sentiments: ')
clf.fit(X_train, Z_train)
z_pred_sentiments_MLP = clf.predict(X_test)
print(z_pred_sentiments_MLP)
print("Accuracy of the dataset using sentiments as a target using Multi-Layered Perceptron is: ",
      metrics.accuracy_score(Z_test, z_pred_sentiments_MLP))
# print("Confusion Matrix for emotions:\n", confusion_matrix(Z_test, z_pred_sentiments_MLP))
performance.write("Confusion Matrix of MLP sentiments:\n")
print(confusion_matrix(Z_test, z_pred_sentiments_MLP), file=performance)
performance.write("Classification report of MLP sentiments: \n")
print(classification_report(Z_test, z_pred_sentiments_MLP), file=performance)
print('----------------------------------------------------')
print('Multi-Layered Perceptron using GridSearchCV: ')

print('Multi-layered perceptron for emotions:')
from sklearn.model_selection import GridSearchCV

parameter_space = {'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
                   'activation': ['logistic', 'tanh', 'relu', 'identity'],
                   'solver': ['adam', 'sgd']}
mlf = GridSearchCV(clf, parameter_space)
mlf.fit(X_train, Y_train)
y_pred_TopMLP_emotions = mlf.predict(X_test)
print("Accuracy of the dataset using emotions as a target using Multi-Layered Perceptron is gridsearchCV: ",
      metrics.accuracy_score(Y_test, y_pred_TopMLP_emotions))
performance.write("Confusion Matrix of ToP MLP emotions:\n")
print(confusion_matrix(Y_test, y_pred_TopMLP_emotions), file=performance)

performance.write("Classification report of Top MLP sentiments: \n")
print(classification_report(Y_test, y_pred_TopMLP_emotions), file=performance)

print('----------------------------------------------------')

print('Multi-layered perceptron for sentiments:')
mlf.fit(X_train, Z_train)
z_pred_sentiments_TopMLP = mlf.predict(X_test)
print("Accuracy of the dataset using sentiments as a target using Multi-Layered Perceptron is gridsearchCV: ",
      metrics.accuracy_score(Z_test, z_pred_sentiments_TopMLP))
performance.write("Confusion Matrix of Top MLP sentiments:\n")
print(confusion_matrix(Z_test, z_pred_sentiments_TopMLP), file=performance)
performance.write("Classification report of Top MLP sentiments: \n")
print(classification_report(Z_test, z_pred_sentiments_TopMLP), file=performance)
print('----------------------------------------------------')

# 2.4
# Creation of a txt file called "precision" to write in it all the results of our analysis' results:
# Implementation of Confusion Matrix:
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_MNB_emotions))
# precision = open('precision.txt','x')

# 2.5
# Use tf-df instead of work frequencies, redo part 2.3
tfidfVectorizer = TfidfVectorizer()
X = tfidfVectorizer.fit_transform(dataSets)
# Multinomial Naive-Bayes for Emotions:
print('----------------------------------------------------')
print('Multinomial Naive-Bayes for emotions using :')
Y = vectorizer.fit_transform(redditData[1])  # Y is the emotions label.
from sklearn.preprocessing import LabelEncoder



performance.close()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
