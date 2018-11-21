# EmotionAnalysis

This repository contains codes of various machine learning approaches for text classification. 
For text classification task, ISEAR (International Survey on Emotion Antecedents and Reactions) (Dataset) [https://github.com/sinmaniphel/py_isear_dataset] is used,
which comprises of 7566 sentences and seven emotion categories including : Anger, Disgust, Fear, Guilt, Joy, Sadness, Shame.
While comparing different algorithms, LSTM turned out to be most effective.

Comparision table :

| Algorithm     | accuracy (%) |
| ------------- | ------------- |
| Naive Bayes with Count vector features               |  54.29  |
| Naive Bayes with with Wordlevel tf-idf features      |  54.16  |
| Logistic regression with Count vector features       |  57.00  |
| Logistic regression with Wordlevel tf-idf features   |  57.00  |
| SVM with Count vector features                       |  54.16  |
| SVM with Wordlevel tf-idf features                   |  58.05  |
| Random forest with Count vector features             |  55.68  |
| Random forest with Wordlevel tf-idf features         |  54.35  |
| ANN without pre-trained embedding matrix             |   |
| ANN with pre-trained embedding matrix (glove)        |   |
| CNN with pre-trained embedding matrix  (glove)       |  55.28  |
| RNN(LSTM) with pre-trained embedding matrix          |  64.6   |


The repository also contained cleaned dataset file in .csv format.
