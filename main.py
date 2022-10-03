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
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from collections import Counter
from sklearn import metrics
#np.set_printoptions(threshold=sys.maxsize)

from sklearn.feature_extraction import DictVectorizer
# from pandas.io import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

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
# Processing the dataset (idea 1) :
#######################################################
#corpus = np.array(loadedData)  # This returns a multidimensional array.
#flatten_array = corpus.flatten()  # We need to transform our array to a one dimensional array.
#vectorizer = CountVectorizer()  # Can we use DictVectorizer instead of count ?
#X = vectorizer.fit_transform(flatten_array)
#X.toarray()
#print('The total number of tokens in the dataset is:', len(vectorizer.get_feature_names_out()))
# Processing the dataset(idea to keep )
######################################################
#print(redditData)
dataSets = redditData[0]
vectorizer = CountVectorizer()
X=vectorizer.fit_transform(dataSets)
#print(vectorizer.get_feature_names_out())
print("Total Number of Tokens: ", len(vectorizer.get_feature_names_out()))
#print(X.toarray()) why is the output not showing !!

#Multinomial Naive-Bayes for Emotions:
print('----------------------------------------------------')
print('Multinomial Naive-Bayes for emotions:')
Y = vectorizer.fit_transform(redditData[1]) # Y is the emotions label.
#print(vectorizer.get_feature_names_out())
#print("---------------------------")
#print(Y)
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()

Y=enc.fit_transform(redditData[1])
#print(Y)
ndf = redditData
ndf[1]= np.transpose(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X_train.toarray())
print(len(X_train.toarray()))

mnb = MultinomialNB()
mnb.fit(X_train,Y_train)
y_pred = mnb.predict(X_test)
print(y_pred)
print("Accuracy of the dataset using emotions as a target using Multinomial Naive-Bayes is: ",metrics.accuracy_score(Y_test,y_pred))

#Multinomial Naiv-Bayes for sentiments:
print('----------------------------------------------------')
print('Multinomial Naive-Bayes for sentiments:')
Z=enc.fit_transform(redditData[2])
ndf=redditData
ndf[2]=np.transpose(Z)
X_train, X_test, Z_train, Z_test = train_test_split(X, Z, test_size=0.2)
mnb.fit(X_train,Z_train)
z_pred=mnb.predict(X_test)
print(z_pred)
print("Accuracy of the dataset using sentiments as a target using Multinomial Naive-Bayes is: ",metrics.accuracy_score(Z_test,z_pred))

print('----------------------------------------------------')
print('Multi-Layered Perceptron for emotions: ')
# Import MLPClassifer
from sklearn.neural_network import MLPClassifier
clf=MLPClassifier()
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
print(y_pred)
print("Accuracy of the dataset using emotions as a target using Multi-Layered Perceptron is: ",metrics.accuracy_score(Y_test,y_pred))
print('----------------------------------------------------')
print('Multi-Layered Perceptron for sentiments: ')
clf.fit(X_train,Z_train)
z_pred=clf.predict(X_test)
print(z_pred)
print("Accuracy of the dataset using sentiments as a target using Multi-Layered Perceptron is: ",metrics.accuracy_score(Z_test,z_pred))
print('----------------------------------------------------')
print('Multi-Layered Perceptron using GridSearchCV: ')




#print(ndf)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
