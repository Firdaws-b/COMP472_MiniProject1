import numpy as np
import json
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from collections import Counter
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from statistics import mean
import gensim.downloader as api
performance = open('performance', 'w')
performance.write("Part 2.4 of the mini project \n")
# precision.write("Emotions: ")
# precision.write("Confusion Matrix:")
# precision.write("Multi-Layered Perceptron Classifier with default parameters, except the number of epochs:  \n")
# precision.write("Precision: \n")
# precision.write("Multi-Layered Perceptron Classifier Using GridSearchCv: \n")


#pathToFile = "/Users/firdawsbouzeghaya/Desktop/geomotions_test-1.json"
pathToFile = "/Users/firdawsbouzeghaya/Downloads/goemotions (1).json"

redditData = pd.read_json(pathToFile)
file = open(pathToFile)
loadedData = json.load(file)
redditPosts = np.array(loadedData)


sentimentClasses = ['positive', 'neutral', 'negative', 'ambiguous']
myRepeatedEmotionsList = redditData[1]  # We first need to get the list of emotions from the data set

# Then we will need to remove the repeated values and convert the list back.

myUnrepeatedEmotionsClassesList = dict.fromkeys(myRepeatedEmotionsList)

myRepeatedEmotionsListSorted = np.sort(myRepeatedEmotionsList)  # Used to sort the emotions' list by alphabetical order

#myUnrepeatedEmotionsClassesList = np.unique(
   # myRepeatedEmotionsListSorted).tolist()  # used to get the labels of the pie chart.

repeatedEmotions = dict(
    Counter(myRepeatedEmotionsListSorted))  # Get the number of each type of emotions for every post.

numberOfPositiveSentiments = redditData[2].value_counts()['positive']
numberOfNegativeSentiments = redditData[2].value_counts()['negative']
numberOfAmbiguousSentiments = redditData[2].value_counts()['ambiguous']
numberOfNeutralSentiments = redditData[2].value_counts()['neutral']


##################################################################
##### This part is word embeddings.

text = redditPosts[:,0]
tokenized_text = [word_tokenize(post) for post in text]
number_of_tokens=len(sum(tokenized_text,[]))
print('Total Number of tokens using tokenizer from nltk 2nd option: ',len(sum(tokenized_text,[])))
# Loading the word2vec pretrained embedding model
#model = api.load('word2vec-google-news-300')
model_vector = api.load('word2vec-google-news-300')



words_vectors = []
for sample in tokenized_text:
   tokens_vec=[model_vector[word] for word in sample if word in model_vector]
   words_vectors.append(tokens_vec)

average_embeddings_posts=[]
for word_vector_of_post in words_vectors:
    for i in range(0, len(word_vector_of_post)):
            #for x in word_vector_of_post[i]:
              # sum = sum + x
            average=np.array(word_vector_of_post[i])
    average_embeddings_posts.append(average)

np.save('embeddings.npy',average_embeddings_posts) # save all the embeddings to a numpy file.
x = np.load('embeddings.npy')
0

           # average_embeddings_posts.append(np.array(word_vector_of_post[i]))
              #  average_embeddings_posts.append(sum)
           # np.array(word_vector_of_post[i]))
       # average_embeddings_token.append(sum(word_vector_of_post[i]) / len(word_vector_of_post[i]))
    #averageEmbedding_of_each_token=[]
    #average_embeddings_posts

average_embeddings_token
#for i in range(0,len(word_vector_of_post)):
 #   average_embeddings_token.append(sum(word_vector_of_post[i])/len(word_vector_of_post[i]))





#for token in tokenized_text:
 #   if(model_vector.most_similar(token)):
        #store the
  #      print("It is similar")
        #list_embeddings_value.append(embedding)

model_vector.most_similar('love')
# Convert tokens into a list of integers
#tokens_to_numbers = word_tokenize.convert_tokens_to_ids(tokenized_text)

#word_vectors = model.wv     # loading the vectors
#word_vectors.save("word2vec.wordvectors")
#print(word_vectors)

#post = gensim.models.word2vec.LineSentences(tokenized_text)
#model.train(posts)
#X=list(model.wv.vcoba)


#corpus = np.array(loadedData)  # This returns a multidimensional array.
#data=np.array2string(corpus)
#print('Total Number of tokens using tokenizer from nltk 2nd option: ',len(word_tokenize(data)))

##############################################################################################


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


# Multinomial Naive-Bayes for Emotions:
print('----------------------------------------------------')
print('Multinomial Naive-Bayes for emotions:')
Y = vectorizer.fit_transform(redditData[1])  # Y is the emotions label.
enc = LabelEncoder()

Y = enc.fit_transform(redditData[1])
ndf = redditData
ndf[1] = np.transpose(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X_train.toarray())
print(len(X_train.toarray()))

##################################### 2.3.1
# Multinomial Naive-Bayes for Emotions:
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
#################### 2.3.3 MLP
# Emotions MLP
print('Multi-Layered Perceptron for emotions: ')
# Import MLPClassifer
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(max_iter=1)
clf.fit(X_train, Y_train)
y_pred_MLP_emotions = clf.predict(X_test)
print("Accuracy of the dataset using emotions as a target using Multi-Layered Perceptron is: ",
      metrics.accuracy_score(Y_test, y_pred_MLP_emotions))
# Implementation of Confusion Matrix of MLP:
performance.write("Confusion Matrix of MLP emotions:\n")
print(confusion_matrix(Y_test, y_pred_MLP_emotions, ), file=performance)
performance.write("Classification report of MLP emotions: \n")
print(classification_report(Y_test, y_pred_MLP_emotions), file=performance)
print('----------------------------------------------------')

# Sentiments Top MLP
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
#################### 2.3.6 TOP MLP
# Emotions Top MLP
print('Multi-Layered Perceptron using GridSearchCV: ')
print('Multi-layered perceptron for emotions:')
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
# Sentiments Top MLP
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

# 2.5
# Remove stop words and redo all the steps of  2.3
stopWordVectorizer = CountVectorizer(stop_words='english')
X_stopWord = stopWordVectorizer.fit_transform(dataSets)
X_train_stopWords, X_test_stopWords, Y_train, Y_test = train_test_split(X_stopWord, Y, test_size=0.2)

##################################### 2.3.1
# Multinomial Naive-Bayes for Emotions:
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

# Multinomial Naive-Bayes for sentiments:
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

##################################### 2.3.3 MLP:
# Emotions MLP
print('Multi-Layered Perceptron for emotions with english stop words: ')
clf.fit(X_train_stopWords, Y_train)
y_pred_MLP_emotions_stopWords = clf.predict(X_test_stopWords)
print("Accuracy of the dataset using emotions as a target using Multi-Layered Perceptron is: ",
      metrics.accuracy_score(Y_test, y_pred_MLP_emotions_stopWords))
performance.write("Confusion Matrix of MLP emotions using english stop words:\n")
print(confusion_matrix(Y_test, y_pred_MLP_emotions_stopWords, ), file=performance)
performance.write("Classification report of MLP emotions using english stop words: \n")
print(classification_report(Y_test, y_pred_MLP_emotions_stopWords), file=performance)
print('----------------------------------------------------')
# Sentiments MLP
print('Multi-Layered Perceptron for sentiments with english stop words: ')
clf.fit(X_train_stopWords, Z_train)
z_pred_sentiments_MLP_stopWords = clf.predict(X_test_stopWords)
print(z_pred_sentiments_MLP_stopWords)
print("Accuracy of the dataset using sentiments as a target using Multi-Layered Perceptron is: ",
      metrics.accuracy_score(Z_test, z_pred_sentiments_MLP_stopWords))
performance.write("Confusion Matrix of MLP sentiments:\n")
print(confusion_matrix(Z_test, z_pred_sentiments_MLP_stopWords), file=performance)
performance.write("Classification report of MLP sentiments: \n")
print(classification_report(Z_test, z_pred_sentiments_MLP_stopWords), file=performance)
print('----------------------------------------------------')

#################### 2.3.6 TOP MLP
# Emotions Top MLP
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
# Sentiments Top MLP
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
############################################################
# Part 3: Embeddings as features:
# 3.1 Use gensim.downloader.load to load the word2vec-google-new-300 pre-trained embedding model
import gensim.downloader as api
model=api.load('word2vec-google-news-300')
# 3.2 Use Tokenizer to extract words from the reddit posts and display the number of tokens in the training set


performance.close()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
