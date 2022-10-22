import gensim.downloader as api
from nltk import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import numpy as np
import json
import pandas as pd
import gensim
import nltk
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import tree, metrics, decomposition, svm
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer

pathToFile = "/Users/kawtherbouzeghaya/Downloads/goemotions.json"
redditData = pd.read_json(pathToFile)
file = open(pathToFile)
loadedData = json.load(file)
redditPosts = np.array(loadedData)
sentimentClasses = ['positive', 'neutral', 'negative', 'ambiguous']
# create a file that contains all the information.
performance = open("/Users/kawtherbouzeghaya/Desktop/COMP472_MiniProject1/performance.txt",'w')
myRepeatedEmotionsList = redditData[1]  # We first need to get the list of emotions from the data set
# Then we will need to remove the repeated values and convert the list back.
myUnrepeatedEmotionsClassesList = dict.fromkeys(myRepeatedEmotionsList)
myRepeatedEmotionsListSorted = np.sort(myRepeatedEmotionsList)  # Used to sort the emotions' list by alphabetical order
repeatedEmotions = dict(
    Counter(myRepeatedEmotionsListSorted))  # Get the number of each type of emotions for every post.

numberOfPositiveSentiments = redditData[2].value_counts()['positive']
numberOfNegativeSentiments = redditData[2].value_counts()['negative']
numberOfAmbiguousSentiments = redditData[2].value_counts()['ambiguous']
numberOfNeutralSentiments = redditData[2].value_counts()['neutral']



#---------------------------------------------------------------------------

# Part One
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

# Part Two : Words as Features.------------------------------------------------------
dataSets = redditData[0]
vectorizer = CountVectorizer()
vectorizer.fit_transform(dataSets)
X = vectorizer.fit_transform(dataSets)
print("Total Number of Tokens: ", len(vectorizer.get_feature_names_out()))

# ****************************** 2.3 Train and test: *****************************
# Data sorting
#-------------------------------------------------------------------------
# Emotions:
Y = vectorizer.fit_transform(redditData[1]) # Y is the emotions label
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

# Sentiments:
Z = vectorizer.fit_transform(redditData[2]) # Z is the sentiments label.
print(vectorizer.get_feature_names_out())
print("---------------------------")
print(Z)
print("---------------------------")
print(X)
enc = LabelEncoder()
Z=enc.fit_transform(redditData[2])
print(Z)
ndf = redditData
ndf[2]= np.transpose(Z)
print(ndf)
X_train, X_test, Z_train, Z_test = train_test_split(X, Z, test_size=0.2)
#---------------------------------------------------------------

#--------------------------2.3.1 Base MNB:---------------------
# Emotions:
print("MNB Emotions analysis")
mnb = MultinomialNB()
mnb.fit(X_train, Y_train)
y_pred_MNB_emotions = mnb.predict(X_test)
print(y_pred_MNB_emotions)
print("Accuracy of the dataset using emotions as a target using Multinomial Naive-Bayes is: ",
      metrics.accuracy_score(Y_test, y_pred_MNB_emotions))

# Sentiments:
print("MNB Sentiments analysis")
mnb.fit(X_train, Z_train)
z_pred_sentiments_MNB = mnb.predict(X_test)
print(z_pred_sentiments_MNB)
print("Accuracy of the dataset using sentiments as a target using Multinomial Naive-Bayes is: ",
      metrics.accuracy_score(Z_test, z_pred_sentiments_MNB))
#---------------------------------------------------------------

#---------------------------2.3.2 Base DT:----------------------
# Emotions:
print("------------------ Emotions Decision Tree analysis------------")
emotionsDecisionTree = tree.DecisionTreeClassifier()
emotionsDecisionTree.fit(X_train,Y_train)
emotionsBaseDecisionTrePrediction = emotionsDecisionTree.predict(X_test)
print("Emotion's Accuracy(without criterion)  :", metrics.accuracy_score(Y_test, emotionsBaseDecisionTrePrediction))

# Sentiments:
print("-------- Sentiments decision Tree analysis")
sentimentDecisionTree = tree.DecisionTreeClassifier()
sentimentDecisionTree.fit(X_train, Z_train)
sentimentBaseDecisionTreePrediction = sentimentDecisionTree.predict(X_test)
print("Sentiments' Accuracy(without criterion)  :", metrics.accuracy_score(Z_test, sentimentBaseDecisionTreePrediction))
#---------------------------------------------------------------

#---------------------------2.3.3 Base MLP:----------------------
# Emotions:
print('Multi-Layered Perceptron for emotions: ')
clf = MLPClassifier(max_iter=1)
clf.fit(X_train, Y_train)
y_pred_MLP_emotions = clf.predict(X_test)
print("Accuracy of the dataset using emotions as a target using Multi-Layered Perceptron is: ",
      metrics.accuracy_score(Y_test, y_pred_MLP_emotions))

# Sentiments:
print('Multi-Layered Perceptron for sentiments: ')
clf.fit(X_train, Z_train)
z_pred_sentiments_MLP = clf.predict(X_test)
print(z_pred_sentiments_MLP)
print("Accuracy of the dataset using sentiments as a target using Multi-Layered Perceptron is: ",
      metrics.accuracy_score(Z_test, z_pred_sentiments_MLP))
#---------------------------------------------------------------

#---------------------------2.3.4 TOP MNB:----------------------
# Emotions:
print("------- TOP NB hyper parameters for emotions ---------")
parametersGrid = {'alpha': [0.5, 0, 2, 1, 0.75]}
grida = GridSearchCV(mnb, param_grid=parametersGrid, cv=5)
grida.fit(X_train, Y_train)
emotionsTopMNBprediction = grida.predict(X_test)
print("Emotion MNB best score is ", grida.best_score_)
print("Emotion MNB best hyper parameter is ", grida.best_params_)

# Sentiments:
print("------- TOP NB hyper parameters for emotions ---------")
parametersGrid = {'alpha': [0.5, 0, 2, 1, 0.75]}
gridb = GridSearchCV(mnb, param_grid=parametersGrid, cv=5)
gridb.fit(X_train, Z_train)
sentimentTopMNBprediction = gridb.predict(X_test)
print("Emotion MNB best score is ", gridb.best_score_)
print("Emotion MNB best hyper parameter is ", gridb.best_params_)
#---------------------------------------------------------------

#---------------------------2.3.5 TOP DT------------------------
print(" Decision Tree analysis (Hyper parameters)using Grid search")

# Setting up our tuning parameters.
parametersGrid = { 'criterion': ['entropy','gini'],
                   'max_depth':[4,6],
                   'min_samples_split':[2,3,6]
                  }
# Emotions:
print("------Emotions Decision Tree analysis (hyper parameters)--------")
grid = GridSearchCV(emotionsDecisionTree, param_grid=parametersGrid, cv=5)
grid.fit(X_train, Y_train)
emotionsTopDecisionTreePrediction = grid.predict(X_test)
print("Emotion DT best score is ", grid.best_score_)
print("Emotion DT best hyper parameters are ", grid.best_params_)

# Sentiments:
print("------Sentimenet Decision Tree analysis (hyper parameters)--------")
grid2 = GridSearchCV(sentimentDecisionTree, param_grid=parametersGrid,cv=5)
grid2.fit(X_train,Z_train)
sentimentTopDecisionTreePrediction = grid2.predict(X_test)
print("Sentiment DT best score is ", grid2.best_score_)
print("Sentiment DT best hyper parameters are ", grid2.best_params_)
#---------------------------------------------------------------

#---------------------------2.3.6 TOP MLP-----------------------
# Emotions:
print('Multi-Layered Perceptron using GridSearchCV: ')
print('Multi-layered perceptron for emotions:')
parameter_space = {'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
                   'activation': ['logistic', 'tanh', 'relu', 'identity'],
                   'solver': ['adam', 'sgd']}
mlf = GridSearchCV(clf, parameter_space)
mlf.fit(X_train, Y_train)
y_pred_TopMLP_emotions = mlf.predict(X_test)
print("Accuracy of the dataset using emotions as a target using top Multi-Layered Perceptron in gridsearchCV: ",
      metrics.accuracy_score(Y_test, y_pred_TopMLP_emotions))
# Sentiments:
print('Multi-layered perceptron for sentiments:')
mlf.fit(X_train, Z_train)
z_pred_sentiments_TopMLP = mlf.predict(X_test)
print("Accuracy of the dataset using sentiments as a target using top Multi-Layered Perceptron in gridsearchCV: ",
      metrics.accuracy_score(Z_test, z_pred_sentiments_TopMLP))
#---------------------------------------------------------------

# ****************************** 2.4 Storing data in a file *****************************
#---------------------------2.4.1 MNB-----------------------
performance.write("Multinomial Naive-Bayes analysis \n")
# Emotions:
performance.write("Accuracy of the dataset using emotions as a target using Multinomial Naive-Bayes is: ")
performance.write(str(metrics.accuracy_score(Y_test, y_pred_MNB_emotions)))
performance.write("\n")
performance.write("Confusion Matrix of MNB emotions:\n")
print(confusion_matrix(Y_test, y_pred_MNB_emotions), file=performance)
performance.write("Classification report of MNB emotions: \n")
print(classification_report(Y_test, y_pred_MNB_emotions), file=performance)
performance.write("\n")
performance.write("Top MNB confusion matrix:\n ")
print(confusion_matrix(Y_test,emotionsTopMNBprediction),file = performance)
performance.write("\n")
performance.write("Top MNB Classification report:\n")
print(classification_report(Y_test,emotionsTopMNBprediction), file=performance)
performance.write("\n")
performance.write("---------------------------------------------------------------------")
performance.write("\n")

# Sentiments:
performance.write("---------------------------------------------------------------\n")
performance.write("Accuracy of the dataset using sentiments as a target using Multinomial Naive-Bayes is: ")
performance.write((str(metrics.accuracy_score(Z_test,z_pred_sentiments_MNB))))
performance.write("\n")
performance.write("Confusion Matrix of MNB sentiments:\n")
print(confusion_matrix(Z_test, z_pred_sentiments_MNB), file=performance)
performance.write("Classification report of MNB sentiments: \n")
print(classification_report(Z_test, z_pred_sentiments_MNB), file=performance)
performance.write("\n")
performance.write("Base DT Classification report:\n")
print(classification_report(Z_test,sentimentTopMNBprediction),file = performance)
performance.write("\n")
performance.write("Top DT confusion matrix\n")
print(confusion_matrix(Z_test,sentimentTopMNBprediction),file = performance)
performance.write("\n")
performance.write("Top DT Classification report:\n")
print(classification_report(Z_test,sentimentTopMNBprediction), file=performance)
performance.write("\n")
performance.write("---------------------------------------------------------------------")
#---------------------------------------------------------------

#---------------------------2.4.2  DT:--------------------------
performance.write("---------------------------------------------------------------\n")
performance.write("Decision Tree analysis: \n")
# Emotions:
performance.write("Accuracy(without criterion) :")
performance.write(str(metrics.accuracy_score(Y_test, emotionsBaseDecisionTrePrediction)))
performance.write("\n")
performance.write("Best hyper parameters ")
performance.write(str(grid.best_params_))
performance.write("\n")
performance.write("Best score ")
performance.write(str(grid.best_score_ ))
performance.write("\n")
performance.write("Base DT confusion matrix:\n ")
print(confusion_matrix(Y_test,emotionsBaseDecisionTrePrediction),file = performance)
performance.write("\n")
performance.write("Base DT Classification report:\n")
print(classification_report(Y_test,emotionsBaseDecisionTrePrediction),file = performance)
performance.write("\n")
performance.write("Top DT confusion matrix:\n ")
print(confusion_matrix(Y_test,emotionsTopDecisionTreePrediction),file = performance)
performance.write("\n")
performance.write("Top DT Classification report:\n")
print(classification_report(Y_test,emotionsTopDecisionTreePrediction), file=performance)
performance.write("\n")
performance.write("---------------------------------------------------------------------")
performance.write("\n")

# Sentiments:
performance.write("Sentiments Decision Tree:")
performance.write("\n")
performance.write("Accuracy(without criterion) : ")
performance.write(str(metrics.accuracy_score(Z_test, sentimentBaseDecisionTreePrediction)))
performance.write("\n")
performance.write("Best hyper parameters" ),
performance.write(str(grid2.best_params_))
performance.write("\n")
performance.write("Best Score "),
performance.write(str(grid2.best_score_))
performance.write("\n")
performance.write("Base DT confusion matrix\n")
print(confusion_matrix(Z_test,sentimentBaseDecisionTreePrediction),file = performance)
performance.write("\n")
performance.write("Base DT Classification report:\n")
print(classification_report(Z_test,sentimentBaseDecisionTreePrediction),file = performance)
performance.write("\n")
performance.write("Top DT confusion matrix\n")
print(confusion_matrix(Z_test,sentimentTopDecisionTreePrediction),file = performance)
performance.write("\n")
performance.write("Top DT Classification report:\n")
print(classification_report(Z_test,sentimentTopDecisionTreePrediction), file=performance)
performance.write("\n")
performance.write("---------------------------------------------------------------------")
#---------------------------------------------------------------

#---------------------------2.4.3 MLP:--------------------------
performance.write("---------------------------------------------------------------\n")
performance.write("Multi Layered Perceptron analysis: \n")
# Emotions:
performance.write("Accuracy of the dataset using emotions as a target using Base Multi-Layered Perceptron is: ") # Base MLP
performance.write(str( metrics.accuracy_score(Y_test, y_pred_MLP_emotions)))
performance.write("\n")
performance.write("Accuracy of the dataset using emotions as a target using top Multi-Layered Perceptron in gridsearchCV: ") # Top MLP
performance.write(str(metrics.accuracy_score(Y_test, y_pred_TopMLP_emotions)))
performance.write("\n")
# Implementation of Confusion Matrix of MLP:base
performance.write("Confusion Matrix of base MLP emotions:\n")
print(confusion_matrix(Y_test, y_pred_MLP_emotions, ), file=performance)
# Classification report.
performance.write("Classification report of base MLP emotions: \n")
print(classification_report(Y_test, y_pred_MLP_emotions), file=performance)
performance.write("\n")
# Implementation of the confusion matrix of top MLP.
performance.write("Confusion Matrix of ToP MLP emotions:\n")
print(confusion_matrix(Y_test, y_pred_TopMLP_emotions), file=performance)
# Implementation of the classification report
performance.write("Classification report of Top MLP emotions: \n")
print(classification_report(Y_test, y_pred_TopMLP_emotions), file=performance)

# Sentiments:
performance.write("Accuracy of the dataset using sentiments as a target using base Multi-Layered Perceptron is: ")
performance.write(str(metrics.accuracy_score(Z_test, z_pred_sentiments_MLP)))
performance.write("\n")
performance.write("Accuracy of the dataset using emotions as a target using top Multi-Layered Perceptron in gridsearchCV:") # Top MLP
performance.write(str(metrics.accuracy_score(Z_test, z_pred_sentiments_TopMLP)))
performance.write("\n")
performance.write("Confusion Matrix of base MLP sentiments:\n")
print(confusion_matrix(Z_test, z_pred_sentiments_MLP), file=performance)
performance.write("Classification report of base MLP sentiments: \n")
print(classification_report(Z_test, z_pred_sentiments_MLP), file=performance)
# Confusion matrix of top mlp
performance.write("Confusion Matrix of Top MLP sentiments:\n")
print(confusion_matrix(Z_test, z_pred_sentiments_TopMLP), file=performance)
# Classification report of top mlp
performance.write("Classification report of Top MLP sentiments: \n")
print(classification_report(Z_test, z_pred_sentiments_TopMLP), file=performance)
print("\n")
#---------------------------------------------------------------


# ****************************** 2.5 Using English Stop words *****************************
performance.write("English stop words analysis")
stopWordVectorizer = CountVectorizer(stop_words='english')
X_stopWord = stopWordVectorizer.fit_transform(dataSets)
X_train_stopWords, X_test_stopWords, Y_train, Y_test = train_test_split(X_stopWord,Y,test_size=0.2)

#---------------------------2.5.1 MNB:--------------------------

# Emotions:
# Base
print('----------------------------------------------------')
print('Multinomial Naive-Bayes for emotions using english stop words:')
mnb.fit(X_train_stopWords, Y_train)
y_pred_stopWords_MNB_emotions = mnb.predict(X_test_stopWords)
print(y_pred_stopWords_MNB_emotions)
print("Accuracy of the dataset using emotions as a target using Multinomial Naive-Bayes is: ",
      metrics.accuracy_score(Y_test, y_pred_stopWords_MNB_emotions))
performance.write("Confusion Matrix of emotions using MNB with english stop words: \n")
performance.write("Confusion Matrix of MNB emotions:\n")
print(confusion_matrix(Y_test, y_pred_stopWords_MNB_emotions), file=performance)
performance.write("Classification report of MNB emotions with english stop words: \n")
print(classification_report(Y_test, y_pred_stopWords_MNB_emotions), file=performance)

#Top
performance.write("Emotions Top MNB: \n")
print("Top MNB for emotions using english stop words: ")
grida.fit(X_train_stopWords,Y_train)
emotionsTopMNBStopWords = grida.predict(X_test_stopWords)
print("Accuracy of the dataset using emotions as a target and Top MNB is ", metrics.accuracy_score(Y_test,emotionsTopMNBStopWords))
performance.write("Confusion Matrix of TMNB emotions using english stop words: \n")
print(confusion_matrix(Y_test, emotionsTopMNBStopWords), file=performance)
performance.write("Classification report for emotions using english stop words:\n")
print(classification_report(Y_test,emotionsTopMNBStopWords), file=performance)
performance.write("\n")
performance.write("----------------------------------------------------------------------")

# Sentiments:
# Base
print('----------------------------------------------------')
print('Multinomial Naive-Bayes for sentiments:')
mnb.fit(X_train_stopWords,Y_train)
y_pred_stopWords_MNB_sentiments = mnb.predict(X_test_stopWords)
print(y_pred_stopWords_MNB_sentiments)
print("Accuracy of the dataset using sentiments as a target using Multinomial Naive-Bayes is: ",
      metrics.accuracy_score(Y_test, y_pred_stopWords_MNB_sentiments))
performance.write("Confusion Matrix of sentiments using MNB with english stop words: \n")
performance.write("Confusion Matrix of MNB sentiments:\n")
print(confusion_matrix(Y_test, y_pred_stopWords_MNB_sentiments), file=performance)
performance.write("Classification report of MNB sentiments with english stop words: \n")
print(classification_report(Y_test, y_pred_stopWords_MNB_sentiments), file=performance)
print('----------------------------------------------------')

# Top
performance.write("Sentiments Top MNB")
print("Top MNB for sentiments using english stop words: ")
gridb.fit(X_train_stopWords,Z_train)
sentimentTopMNBStopWords = gridb.predict(X_test_stopWords)
print("Accuracy of the dataset using sentiments as a target and Top MNB is", metrics.accuracy_score(Z_test,sentimentTopMNBStopWords))
performance.write("Confusion Matrix of TMNB sentiments using english stop words: \n")
print(confusion_matrix(Z_test,sentimentTopMNBStopWords), file=performance)
performance.write("Classification report for sentiments using english stop words:\n")
print(classification_report(Z_test,sentimentTopMNBStopWords), file=performance)
performance.write("\n")
performance.write("--------------------------------------------------------------------")
#---------------------------------------------------------------

#---------------------------2.5.2 DT:--------------------------

# Emotions:
performance.write("Emotions Base Decision Tree: \n")
# Base
print("Base Decision Tree for emotions using english stop words: ")
emotionsDecisionTree.fit(X_train_stopWords,Y_train)
emotionsBaseDecisionTreeStopWords = emotionsDecisionTree.predict(X_test_stopWords)
print("Accuracy of the dataset using emotions as a target and Base DT is ", metrics.accuracy_score(Y_test,emotionsBaseDecisionTreeStopWords))
performance.write("Confusion Matrix of DT emotions using english stop words: \n")
print(confusion_matrix(Y_test,emotionsBaseDecisionTreeStopWords), file=performance)
performance.write("Classification report for emotions using english stop words:\n")
print(classification_report(Y_test,emotionsBaseDecisionTreeStopWords), file=performance)
performance.write("\n")

# Top:
performance.write("Emotions Top Decision Tree: \n")
print("Top Decision Tree for emotions using english stop words: ")
grid.fit(X_train_stopWords,Y_train)
emotionsTopDecisionTreeStopWords = grid.predict(X_test_stopWords)
print("Accuracy of the dataset using emotions as a target and Top DT is ", metrics.accuracy_score(Y_test,emotionsTopDecisionTreeStopWords))
performance.write("Confusion Matrix of TDT emotions using english stop words: \n")
print(confusion_matrix(Y_test, emotionsTopDecisionTreeStopWords), file=performance)
performance.write("Classification report for emotions using english stop words:\n")
print(classification_report(Y_test,emotionsTopDecisionTreeStopWords), file=performance)
performance.write("\n")
performance.write("----------------------------------------------------------------------")

# Sentiments:
performance.write("Sentiments Base DT: \n")
print("Base Decision Tree for sentiments with english stop words: ")
sentimentDecisionTree.fit(X_train_stopWords,Z_train)
sentimentBaseDecisionTreeStopWords = sentimentDecisionTree.predict(X_test_stopWords)
print("Accuracy of the dataset using sentiments as a target and Base DT is ",metrics.accuracy_score(Z_test,sentimentBaseDecisionTreeStopWords))
performance.write("Confusion Matrix of Base DT sentiments using english stop words :\n")
print(confusion_matrix(Z_test,sentimentBaseDecisionTreeStopWords), file=performance)
performance.write("Classification report for sentiment using english stop words:\n")
print(classification_report(Z_test,sentimentBaseDecisionTreeStopWords), file=performance)
performance.write("\n")
performance.write("----------------------------------------------------------")

performance.write("Sentiments Top DT")
print("Top Decision Tree for sentiments using english stop words: ")
grid2.fit(X_train_stopWords,Z_train)
sentimentTopDecisionTreeStopWords = grid2.predict(X_test_stopWords)
print("Accuracy of the dataset using sentiments as a target and Top DT is", metrics.accuracy_score(Z_test,sentimentTopDecisionTreeStopWords))
performance.write("Confusion Matrix of TDT sentiments using english stop words: \n")
print(confusion_matrix(Z_test,sentimentTopDecisionTreeStopWords), file=performance)
performance.write("Classification report for sentiments using english stop words:\n")
print(classification_report(Z_test,sentimentTopDecisionTreeStopWords), file=performance)
performance.write("\n")
performance.write("--------------------------------------------------------------------")
#---------------------------------------------------------------

#---------------------------2.5.3 MLP:--------------------------

# Base
# Emotions: Base
print('Multi-Layered Perceptron for emotions with english stop words: ')
clf.fit(X_train_stopWords, Y_train)
y_pred_MLP_emotions_stopWords = clf.predict(X_test_stopWords)
print("Accuracy of the dataset using emotions as a target using Base Multi-Layered Perceptron is: ",
      metrics.accuracy_score(Y_test, y_pred_MLP_emotions_stopWords))
performance.write("Confusion Matrix of Base MLP emotions using english stop words:\n")
print(confusion_matrix(Y_test, y_pred_MLP_emotions_stopWords, ), file=performance)
performance.write("Classification report of Base MLP emotions using english stop words: \n")
print(classification_report(Y_test, y_pred_MLP_emotions_stopWords), file=performance)
print('----------------------------------------------------')

# Sentiments :
print('Multi-Layered Perceptron for sentiments with english stop words: ')
clf.fit(X_train_stopWords, Z_train)
z_pred_sentiments_MLP_stopWords = clf.predict(X_test_stopWords)
print(z_pred_sentiments_MLP_stopWords)
print("Accuracy of the dataset using sentiments as a target using Base Multi-Layered Perceptron is: ",
      metrics.accuracy_score(Z_test, z_pred_sentiments_MLP_stopWords))
performance.write("Confusion Matrix of Base MLP sentiments:\n")
print(confusion_matrix(Z_test, z_pred_sentiments_MLP_stopWords), file=performance)
performance.write("Classification report of Base MLP sentiments: \n")
print(classification_report(Z_test, z_pred_sentiments_MLP_stopWords), file=performance)
print('----------------------------------------------------')

#TOP
# Emotions:
print('----------------------------------------------------')
print('Multi-Layered Perceptron using GridSearchCV using english stop words: ')
print('Multi-layered perceptron for emotions using english stop words:')
mlf.fit(X_train_stopWords, Y_train)
y_pred_TopMLP_emotions_stopWords = mlf.predict(X_test_stopWords)
print("Accuracy of the dataset using emotions as a target using Multi-Layered Perceptron in gridsearchCV: ",
      metrics.accuracy_score(Y_test, y_pred_TopMLP_emotions_stopWords))
performance.write("Confusion Matrix of ToP MLP emotions using english stop words:\n")
print(confusion_matrix(Y_test, y_pred_TopMLP_emotions_stopWords), file=performance)
performance.write("Classification report of Top MLP sentiments: \n")
print(classification_report(Y_test, y_pred_TopMLP_emotions_stopWords), file=performance)
print('----------------------------------------------------')

# Sentiments
print('Top Multi-layered perceptron for sentiments using english stop words :')
mlf.fit(X_train_stopWords, Z_train)
z_pred_sentiments_TopMLP_stopWords = mlf.predict(X_test_stopWords)
print("Accuracy of the dataset using sentiments as a target using Multi-Layered Perceptron with gridsearchCV: ",
      metrics.accuracy_score(Z_test, z_pred_sentiments_TopMLP_stopWords))
performance.write("Confusion Matrix of Top MLP sentiments:\n")
print(confusion_matrix(Z_test, z_pred_sentiments_TopMLP_stopWords), file=performance)
performance.write("Classification report of Top MLP sentiments: \n")
print(classification_report(Z_test, z_pred_sentiments_TopMLP_stopWords), file=performance)
print('----------------------------------------------------')

# Part Three: Embeddings as Features:

# 3.2
text = redditPosts[:, 0]
tokenized_text = [word_tokenize(post) for post in text]
number_of_tokens = len(sum(tokenized_text, []))
print('Total Number of tokens using tokenizer from nltk 2nd option: ', len(sum(tokenized_text, [])))
# Loading the word2vec pretrained embedding model
model_vector = api.load('word2vec-google-news-300')

# 3.3 & 3.4  compute the embeddings and display them.
words_vectors = []
count_number_of_words_found = 0
for sample in tokenized_text:
    tokens_vec = [model_vector[word] for word in sample if word in model_vector]
    for word in sample:
        if word in model_vector:
            count_number_of_words_found = count_number_of_words_found + 1
    words_vectors.append(tokens_vec)

average_embeddings_posts = []
for word_vector_of_post in words_vectors:
    for i in range(0, len(word_vector_of_post)):
        average = np.array(word_vector_of_post[i])
    average_embeddings_posts.append(average)

np.save('embeddings.npy', average_embeddings_posts)  # save all the embeddings to a numpy file.
x = np.load('embeddings.npy')
overall_hit = (count_number_of_words_found / number_of_tokens)*100
print(
    "Overall hit rates of the training and test sets (% of words in the reddit post for which an embedding is found "
    "in Word2Vec:  ", overall_hit)

# 3.5 and 3.7 Train a base MLP
# Emotions
X_embeddings_train, X_embeddings_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.2)
clf = MLPClassifier(max_iter=1)
clf.fit(X_embeddings_train, Y_train)
y_pred_MLP_embeddings_emotions = clf.predict(X_embeddings_test)
print("Accuracy of the dataset using emotions as a target using Multi-Layered Perceptron with word embeddings is: ",
      metrics.accuracy_score(Y_test, y_pred_MLP_embeddings_emotions))
performance.write("Confusion Matrix of MLP using word embeddings for emotions:\n")
print(confusion_matrix(Y_test, y_pred_MLP_embeddings_emotions, ), file=performance)
performance.write("Classification report of MLP emotions: \n")
print(classification_report(Y_test, y_pred_MLP_embeddings_emotions), file=performance)

# Sentiments
X_embeddings_train, X_embeddings_test, Z_train, Z_test = train_test_split(x, Z, test_size=0.2)
clf.fit(X_embeddings_train,Z_train)
z_pred_MLP_embeddings_sentiments = clf.predict(X_embeddings_test)
print("Accuracy of the dataset using sentiments as a target using Multi-Layered Perceptron with word embeddings is: ",
      metrics.accuracy_score(Z_test, z_pred_MLP_embeddings_sentiments))
performance.write("Confusion Matrix of MLP using word embeddings for sentiments:\n")
print(confusion_matrix(Z_test, z_pred_MLP_embeddings_sentiments, ), file=performance)
performance.write("Classification report of MLP sentiments: \n")
print(classification_report(Z_test, z_pred_MLP_embeddings_sentiments), file=performance)

# 3.6 Train a Top MLP
# Emotions:
mlf.fit(X_embeddings_train, Y_train)
y_pred_TopMLP_embeddings_emotions = mlf.predict(X_embeddings_test)
print("Accuracy of the dataset using emotions as a target using Top Multi-Layered Perceptron with word embeddings is: ",
      metrics.accuracy_score(Y_test, y_pred_TopMLP_embeddings_emotions))
performance.write("Confusion Matrix of Top MLP using word embeddings for emotions:\n")
print(confusion_matrix(Y_test, y_pred_TopMLP_embeddings_emotions, ), file=performance)
performance.write("Classification report of Top MLP emotions: \n")
print(classification_report(Y_test, y_pred_TopMLP_embeddings_emotions), file=performance)

# Sentiments
mlf.fit(X_embeddings_train, Z_train)
z_pred_TopMLP_embeddings_sentiments = mlf.predict(X_embeddings_test)
print("Accuracy of the dataset using sentiments as a target using Top Multi-Layered Perceptron with word embeddings "
      "is: ",
      metrics.accuracy_score(Y_test, z_pred_TopMLP_embeddings_sentiments))
performance.write("Confusion Matrix of Top MLP using word embeddings for sentiments:\n")
print(confusion_matrix(Z_test, z_pred_TopMLP_embeddings_sentiments, ), file=performance)
performance.write("Classification report of Top MLP sentiments: \n")
print(classification_report(Z_test, z_pred_TopMLP_embeddings_sentiments), file=performance)

# 3.8 Run the best performing model with two other english pretrained models:
# Using glove-wiki-gigaword-50
model_vector_glove_wiki_gigaword_50 = api.load('glove-wiki-gigaword-50')
words_vectors_glove_wiki = []
for sample_2 in tokenized_text:
    tokens_vec_2 = [model_vector_glove_wiki_gigaword_50[word] for word in sample_2 if word in model_vector_glove_wiki_gigaword_50]
    words_vectors_glove_wiki.append(tokens_vec_2)

average_embeddings_posts_model2 = []
for word_vector_of_post_model2 in words_vectors_glove_wiki:
    for i in range(0, len(word_vector_of_post_model2)):
        average = np.array(word_vector_of_post_model2[i])
    average_embeddings_posts_model2.append(average)

np.save('embeddings_glove_wiki.npy', average_embeddings_posts_model2)  # save all the embeddings to a numpy file.
x_model2 = np.load('embeddings_glove_wiki.npy')
# Train MLP:
# emotions:
X_embeddings_2_train, X_embeddings_2_test, Y_train, Y_test = train_test_split(x_model2, Y, test_size=0.2)
clf = MLPClassifier(max_iter=1)
clf.fit(X_embeddings_2_train, Y_train)
y_pred_MLP_embeddings2_emotions = clf.predict(X_embeddings_2_test)
print("Accuracy of the dataset using emotions as a target using Multi-Layered Perceptron with word embeddings model "
      "glove-wiki-gigaword-50 is: ",
      metrics.accuracy_score(Y_test, y_pred_MLP_embeddings2_emotions))
performance.write("Confusion Matrix of MLP using word embeddings pretrained model glove-wiki-gigaword-50 for emotions:\n")
print(confusion_matrix(Y_test, y_pred_MLP_embeddings2_emotions, ), file=performance)
performance.write("Classification report of MLP emotions: \n")
print(classification_report(Y_test, y_pred_MLP_embeddings2_emotions), file=performance)

# sentiments:
X_embeddings_2_train, X_embeddings_2_test, Z_train, Z_test = train_test_split(x_model2, Z, test_size=0.2)
clf.fit(X_embeddings_2_train, Y_train)
z_pred_MLP_embeddings2_sentiments = clf.predict(X_embeddings_2_test)
print("Accuracy of the dataset using sentiments as a target using Multi-Layered Perceptron with word embeddings model "
      "glove-wiki-gigaword-50 is: ",
      metrics.accuracy_score(Z_test, z_pred_MLP_embeddings2_sentiments))
performance.write("Confusion Matrix of MLP using word embeddings pretrained model glove-wiki-gigaword-50 for sentiments:\n")
print(confusion_matrix(Z_test, z_pred_MLP_embeddings2_sentiments, ), file=performance)
performance.write("Classification report of MLP sentiments: \n")
print(classification_report(Z_test, z_pred_MLP_embeddings2_sentiments), file=performance)

##############################################################################################
# Model 3: Using glove-twitter-25
# Getting the embeddings:
model_vector_glove_twitter_25 = api.load('glove-twitter-25')
words_vectors_glove_twitter_25 = []
for sample_3 in tokenized_text:
    tokens_vec_3 = [model_vector_glove_twitter_25[word] for word in sample_3 if word in model_vector_glove_twitter_25]
    words_vectors_glove_twitter_25.append(tokens_vec_3)

average_embeddings_posts_model3 = []
for word_vector_of_post_model3 in words_vectors_glove_twitter_25:
    for i in range(0, len(word_vector_of_post_model3)):
        average = np.array(word_vector_of_post_model3[i])
    average_embeddings_posts_model3.append(average)

np.save('embeddings_glove_twitter.npy', average_embeddings_posts_model3)  # save all the embeddings to a numpy file.
x_model3 = np.load('embeddings_glove_twitter.npy')

# Train the models
# MLP
# emotions:
X_embeddings_3_train, X_embeddings_3_test, Y_train, Y_test = train_test_split(x_model3, Y, test_size=0.2)
clf = MLPClassifier(max_iter=1)
clf.fit(X_embeddings_3_train, Y_train)
y_pred_MLP_embeddings3_emotions = clf.predict(X_embeddings_3_test)
print("Accuracy of the dataset using emotions as a target using Multi-Layered Perceptron with pretrained model "
      "glove-twitter-25 is: ",
      metrics.accuracy_score(Y_test, y_pred_MLP_embeddings2_emotions))
performance.write("Confusion Matrix of MLP using word embeddings with pretrained model glove-twitter-25 for "
                  "emotions:\n")
print(confusion_matrix(Y_test, y_pred_MLP_embeddings3_emotions, ), file=performance)
performance.write("Classification report of MLP emotions: \n")
print(classification_report(Y_test, y_pred_MLP_embeddings3_emotions), file=performance)

# sentiments:
X_embeddings_3_train, X_embeddings_3_test, Z_train, Z_test = train_test_split(x_model3, Z, test_size=0.2)
clf.fit(X_embeddings_3_train, Y_train)
z_pred_MLP_embeddings3_sentiments = clf.predict(X_embeddings_3_test)
print("Accuracy of the dataset using sentiments as a target using Multi-Layered Perceptron with pretrained embeddings model "
      "glove-twitter-25 is: ",
      metrics.accuracy_score(Z_test, z_pred_MLP_embeddings3_sentiments))
performance.write("Confusion Matrix of MLP using pretrained embedding model glove-twitter-25 for sentiments:\n")
print(confusion_matrix(Z_test, z_pred_MLP_embeddings3_sentiments, ), file=performance)
performance.write("Classification report of MLP sentiments: \n")
print(classification_report(Z_test, z_pred_MLP_embeddings3_sentiments), file=performance)

performance.close()
#---------------------------------------------------------------
