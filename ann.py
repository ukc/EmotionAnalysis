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


embedding_dim = 50

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
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




