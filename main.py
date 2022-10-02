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
from matplotlib import pyplot as plt
from collections import Counter
#np.set_printoptions(threshold=sys.maxsize)

from sklearn.feature_extraction import DictVectorizer
# from pandas.io import json
from sklearn.feature_extraction.text import CountVectorizer

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
# Processing the dataset:
##sizeOfTokensInTheDataSet=
# print(redditData)
# dataSets=redditData
# vectorizer=CountVectorizer()
# matrix = vectorizer.fit_transform(dataSets)
# print(myRepeatedEmotionsList)
# print("Tokens: ", vectorizer.vocabulary_)
# print(loadedData)

corpus = np.array(loadedData)  # This returns a multidimensional array.
flatten_array = corpus.flatten()  # We need to transform our array to a one dimensional array.
# vectorizer = CountVectorizer()
vectorizer = CountVectorizer()  # Can we use DictVectorizer instead of count ?
X = vectorizer.fit_transform(flatten_array)

#frequencyTokens = dict(
 #   Counter(flatten_array))
#print(frequencyTokens)
#print(vectorizer.get_feature_names_out())
print(X.toarray())
#dataArray = vectorizer.get_feature_names_out()
#print(dataArray)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
