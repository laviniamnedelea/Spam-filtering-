import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


def get_data(filename):

    df = pd.read_csv(filename, delimiter=',')
    labels = df['type'].values
    mesages = df['text'].values

    # pos_examples = np.sum(labels == 'ham') / labels.shape[0]
    # neg_examples = 1 - pos_examples
    # print(pos_examples)
    shuffle_stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.2) #we will divide our data

    for train_index, test_index in shuffle_stratified.split(mesages, labels):
        msg_train, msg_test = mesages[train_index], mesages[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

    y_train = np.int32(labels_train == 'ham') #we will save our train labels in y_train in binary form
    y_test = np.int32(labels_test == 'ham') #we'll do the same with the test labels


    return msg_train, y_train, msg_test, y_test


msg_train, y_train, msg_test, y_test = get_data('sms_spam.csv')

count_vectorizer = CountVectorizer(lowercase=True, analyzer='word', stop_words='english')

count_vectorizer.fit(msg_train)
X_train = count_vectorizer.transform(msg_train)
X_test = count_vectorizer.transform(msg_test)

model = MultinomialNB(alpha=0.01)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print(model.predict_proba(X_test[0]))
