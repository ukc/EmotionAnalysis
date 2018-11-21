import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer

from keras.models import Sequential
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
import os
import pickle

def train_model(classifier, feature_vector_train, train_label, feature_vector_test, test_label):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, train_label)
    # predict the labels on test dataset
    predictions = classifier.predict(feature_vector_test)
    #print(classification_report(test_label, predictions))
    return accuracy_score(predictions, test_label)


def loadGloveModel(gloveFile):
    glove_model = {} 
    f = open(gloveFile,'r')
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        glove_model[word] = embedding
    with open('model.pickle', 'wb') as handle:
        pickle.dump(glove_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return glove_model


glove_model = {}
# loading pre-traing word2vec
if os.path.isfile('glove_model.pickle'):
	handle = open('glove_model.pickle', 'rb')
	glove_model = pickle.load(handle)
else:
    glove_model = loadGloveModel('../data/glove.6B/glove.6B.300d.txt')


def create_embedding_matrix(glove_model, word_index, embedding_dim):
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word in glove_model:
        vector = glove_model[word]
        if word in word_index:
            idx = word_index[word] 
            embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix


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
X_train, X_test, y_train , y_test = train_test_split(dataDF['text'], dataDF['label'], random_state = 24, test_size = 0.2)


# encode the target variable 
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)
onehotencoder = OneHotEncoder(sparse=False)
y_train = onehotencoder.fit_transform(y_train.reshape(-1, 1))
y_test  = onehotencoder.fit_transform(y_test.reshape(-1, 1))


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(dataDF['text'])

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1 

maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


embedding_dim = 300
embedding_matrix = create_embedding_matrix(glove_model, tokenizer.word_index, embedding_dim)


model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, 
                           weights=[embedding_matrix], 
                           input_length=maxlen, 
                           trainable=True))

model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=20,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))




