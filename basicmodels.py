import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC , SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report

def train_model(classifier, feature_vector_train, train_label, feature_vector_test, test_label):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, train_label)
    # predict the labels on test dataset
    predictions = classifier.predict(feature_vector_test)
    #print(classification_report(test_label, predictions))
    return accuracy_score(predictions, test_label)



# load the dataset
TRAIN_SET_PATH = "../data/real_data.csv"
data = open(TRAIN_SET_PATH).readlines()

labels, texts = [], []
for line in data:
	label, text = line.split("\t")
	labels.append(label.strip().lower())
	texts.append(text.strip().split())


# create a dataframe using texts and lables
dataDF = pd.DataFrame()
dataDF['text'] = texts
dataDF['label'] = labels

print(dataDF.head())
print("total examples %s" % len(labels))

# split the dataset into training and test datasets 
train_x, test_x, train_y, test_y = train_test_split(dataDF['text'], dataDF['label'], random_state = 24)

# label encode the target variable 
encoder = LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)


# create a count vectorizer object 
count_vect = CountVectorizer(analyzer=lambda x: x)
count_vect.fit(dataDF['text'])

# transform the training and test data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xtest_count =  count_vect.transform(test_x)

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer=lambda x: x)
tfidf_vect.fit(dataDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xtest_tfidf =  tfidf_vect.transform(test_x)


'''
MODELS
'''

## Naive Bayes

# Naive Bayes on Count Vectors
accuracy = train_model(MultinomialNB(), xtrain_count, train_y, xtest_count, test_y)
print("NB, Count Vectors: ", accuracy)



# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(MultinomialNB(), xtrain_tfidf, train_y, xtest_tfidf, test_y)
print("NB, WordLevel TF-IDF: ", accuracy)


## Logistic regression

# Logistic regression on Count Vectors
accuracy = train_model(LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state= 24), xtrain_count, train_y, xtest_count, test_y)
print("LR, Count Vectors: ", accuracy)

# Logistic regression on Word Level TF IDF Vectors
accuracy = train_model(LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state= 24), xtrain_tfidf, train_y, xtest_tfidf, test_y)
print("LR, WordLevel TF-IDF: ", accuracy)


## SVM 

# SVM on Count Vectors
accuracy = train_model(LinearSVC(random_state= 24), xtrain_count, train_y, xtest_count, test_y)
print("SVM, Count Vectors: ", accuracy)

# SVM on Ngram Level TF IDF Vectors
accuracy = train_model(LinearSVC(random_state= 24), xtrain_tfidf, train_y, xtest_tfidf, test_y)
print("SVM, WodLevel TF-IDF: ", accuracy)


## Random Forest

# Random Forest on Count Vectors
accuracy = train_model(RandomForestClassifier(n_estimators=150, random_state= 24), xtrain_count, train_y, xtest_count, test_y)
print("RF, Count Vectors: ", accuracy)

# Random Forest on Word Level TF IDF Vectors
accuracy = train_model(RandomForestClassifier(n_estimators=150, random_state= 24), xtrain_tfidf, train_y, xtest_tfidf, test_y)
print("RF, WordLevel TF-IDF: ", accuracy)
