# This is a sample Python script.
from collections import OrderedDict

import numpy as np
import json
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import gensim
import nltk
from matplotlib import pyplot as plt
from collections import Counter

#from pandas.io import json
from sklearn.feature_extraction.text import CountVectorizer


pathToFile = "/Users/kawtherbouzeghaya/Downloads/goemotions.json"

redditData = pd.read_json(pathToFile)
file=open(pathToFile)
loadedData=json.load(file)
sentimentClasses=['positive','neutral','negative','ambiguous']
myRepeatedEmotionsList=redditData[1] #We first need to get the list of emotions from the data set
                                     #Then we will need to remove the repeated values and convert the list back.

myUnrepeatedEmotionsClassesList=dict.fromkeys(myRepeatedEmotionsList)

myRepeatedEmotionsListSorted=np.sort(myRepeatedEmotionsList)#Used to sort the emotions' list by alphabetical order

myUnrepeatedEmotionsClassesList=np.unique(myRepeatedEmotionsListSorted).tolist() #used to get the labels of the pie chart.


repeatedEmotions=dict(Counter(myRepeatedEmotionsListSorted)) #Get the number of each type of emotions for every post.


#At first I created the histogram, but it was too ugly :( so I had to switch to pie charts !
#plt.hist(myRepeatedEmotionsList,bins=28)
#plt.show()


#print(myUnrepeatedEmotionsClassesList)
#print(redditData)
numberOfPositiveSentiments=redditData[2].value_counts()['positive']
numberOfNegativeSentiments=redditData[2].value_counts()['negative']
numberOfAmbiguousSentiments=redditData[2].value_counts()['ambiguous']
numberOfNeutralSentiments=redditData[2].value_counts()['neutral']
#This print is used
#print(repeatedEmotions)
#Labels
sentimentClassesData=[numberOfPositiveSentiments,numberOfNegativeSentiments,numberOfAmbiguousSentiments,numberOfNeutralSentiments]
#Plotting the pie chart of emotions
numberOfEmotions=[]
labelsOfEmotions=[]
for x,y in repeatedEmotions.items():
    labelsOfEmotions.append(x)
    numberOfEmotions.append(y)

emotionsPieChart=plt.figure(figsize=(10,7))
plt.pie(numberOfEmotions,labels=labelsOfEmotions,autopct='%1.0f%%')
plt.title('Reddit Posts Emotions Distribution.')

sentimentPieChart=plt.figure(figsize=(10,7)) #Set the size of the figure
plt.pie(sentimentClassesData,labels=sentimentClasses,autopct='%1.0f%%')
plt.title('Reddit Posts Sentiments Distribution.')
plt.show()

#Part Two:
#Processing the dataset:
##sizeOfTokensInTheDataSet=
print(redditData)
dataSets=redditData[0]
vectorizer=CountVectorizer()
vectorizer.fit(dataSets)
print("Tokens: ", vectorizer.vocabulary_)

# Part Two:
# Processing the dataset:
# sizeOfTokensInTheDataSet=
print(redditData)
dataSets = redditData[0]
vectorizer = CountVectorizer()
vectorizer.fit_transform(dataSets)
print("Tokens: ", vectorizer.vocabulary_)

# Part Three:
# Creating a training dataset
X = vectorizer.fit_transform(dataSets)
print(vectorizer.get_feature_names_out())
print(X.toarray())

Y = vectorizer.fit_transform(redditData[1])
print(vectorizer.get_feature_names_out())

print("---------------------------")
print(Y)
print("---------------------------")
print(X)
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()


Y=enc.fit_transform(redditData[1])
print(Y)

ndf = redditData
ndf[1]= np.transpose(Y)
print(ndf)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

print(X_train.toarray())
print(len(X_train.toarray()))

from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb.fit(X_train,Y_train)
y_pred = mnb.predict(X_test)
print(y_pred)
from sklearn import metrics
print(metrics.accuracy_score(Y_test,y_pred))
