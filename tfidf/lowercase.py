#!/usr/bin/env python3
import pickle

with open('text.pickle', 'rb') as pick:
    docs = pickle.load(pick)

for doc in docs:
    doc['words'] = [word.lower() for word in doc['words']]

with open('text-lower.pickle', 'wb') as pick:
    pickle.dump(docs, pick)


# words = [word for doc in docs for word in doc['words']]
# n_words = len(words)
# print(len(words))
# print(len(set(words)))
