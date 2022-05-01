import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import re
train=pd.read_csv("./corpus/imdb/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
# nltk.download()
fi=open('./corpus/english_stop.txt','r')
txt=fi.readlines()
a=[]
for w in txt:
    w=w.replace('\n','')
    a.append(w)
# print(a)


def review_to_words(review):
    review_text = BeautifulSoup(review).get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stop_words = set(a)
    final_words = [w for w in words if w not in stop_words]
    return (" ".join(final_words))
clean_review = review_to_words(train["review"][1])
# print(clean_review)
num_reviews = len(train["review"])

clean_train_reviews = []

for i in range(0, num_reviews):
    clean_train_reviews.append(review_to_words(train["review"][i]))
    if ((i+1) % 1000 == 0):
        print("Review %d of %d" % (i+1, num_reviews))
vectorizer = CountVectorizer(max_features = 5000)
train_features = vectorizer.fit_transform(clean_train_reviews)
train_features = train_features.toarray()
vocab = vectorizer.get_feature_names()
# dist = np.sum(train_features, axis = 0)
forest = RandomForestClassifier(verbose = 0, random_state = 0)
forest.fit(train_features, train['sentiment'])
test = pd.read_csv("./corpus/imdb/testData.tsv", header=0, delimiter="\t", quoting=3)
print(test.shape)
test.head()
num_test_reviews = len(test)
clean_test_reviews = []

for i in range(0, num_test_reviews):
    clean_test_reviews.append(review_to_words(test["review"][i]))
    if ((i+1) % 5000 == 0):
        print("Review %d of %d" % (i+1, num_test_reviews))
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
result = forest.predict(test_data_features)
output = pd.DataFrame(data={'id':test['id'], 'sentiment':result})
output.to_csv("submission1.csv", index = False, quoting=3)