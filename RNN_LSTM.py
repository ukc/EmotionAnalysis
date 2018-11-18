import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, GRU
from keras.preprocessing.sequence import pad_sequences
import os
import pickle

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV


def loadGloveModel(gloveFile):
    glove_model = {} 
    f = open(gloveFile,'r')
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        #embedding = np.asarray(splitLine[1:], dtype='float32')
        glove_model[word] = embedding
    with open('glove_model.pickle', 'wb') as handle:
        pickle.dump(glove_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return glove_model


glove_model = {}
# loading pre-traing word2vec
if os.path.isfile('glove_model.pickle'):
	handle = open('glove_model.pickle', 'rb')
	glove_model = pickle.load(handle)
else:
    glove_model = loadGloveModel('../data/glove.6B/glove.6B.300d.txt')

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


tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(dataDF['text'])

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1 
print(vocab_size)
maxlen = 100
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)


def create_embedding_matrix(word_index, embedding_dim):
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word in glove_model:
        vector = glove_model[word]
        if word in word_index:
            idx = word_index[word] 
            embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix




def create_model(vocab_size, embedding_dim, maxlen, dropout_val):
	embedding_matrix = create_embedding_matrix(tokenizer.word_index, embedding_dim)
	model = Sequential()
	model.add(layers.Embedding(vocab_size, embedding_dim, 
		                       weights=[embedding_matrix], 
		                       input_length=maxlen, 
		                       trainable=False))
	#model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
	model.add(LSTM(100, dropout= dropout_val, recurrent_dropout=dropout_val))
	#model.add(layers.Dense(10, activation='relu'))
	model.add(Dense(7, activation='softmax'))
	model.compile(optimizer='adam',
		          loss='categorical_crossentropy',
		          metrics=['accuracy'])
	return model


model = create_model(vocab_size, 150, maxlen, 0.4)
#model.summary()

param_grid = dict(vocab_size=[vocab_size], 
                  embedding_dim=[300, 150, 100],
                  maxlen=[maxlen],
				  dropout_val = [0.2,0.4,0.6])

checkpoint_callback = ModelCheckpoint('emotional_rnn.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_acc',
                              min_delta=0,
                              patience=100,
                              verbose=0, mode='auto')


'''model = KerasClassifier(build_fn=create_model,
                        epochs=100, batch_size=128,
                        verbose=2) 

grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=4, verbose=1, n_iter=5)
grid_result = grid.fit(X_train, y_train)

# Evaluate testing set
test_accuracy = grid.score(X_test, y_test)


with open('LSTM_grid_search.txt' , 'w') as f:
    s = ('Best Accuracy : {:.4f}\n {}\n Test Accuracy : {:.4f}\n')
    output_string = s.format( grid_result.best_score_, grid_result.best_params_, test_accuracy)
    print(output_string)
    f.write(output_string)'''


history = model.fit(X_train, y_train,
                    epochs= 200,
                    verbose= 2,
                    validation_data=(X_test, y_test),
                    batch_size= 128, callbacks = [checkpoint_callback, early_stopping])

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))




