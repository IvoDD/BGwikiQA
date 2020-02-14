#!/usr/bin/env python3
import json
import time
import pickle
import stanfordnlp
nlp = stanfordnlp.Pipeline(processors='tokenize,pos', lang='bg', use_gpu=True)

with open('text.json') as dataset:
    docs = [json.loads(line) for line in dataset]

def text_to_words(text):
    sentences = nlp(text).sentences
    tokens = [token for sentence in sentences for token in sentence.tokens]
    words = [word for token in tokens for word in token.words]
    words = [word for word in words if word.upos != 'PUNCT']
    lemmas = [] # lemmas = [word.lemma for word in words]
    words = [word.text for word in words]
    return words, lemmas

c_bytes = 0
n_bytes = sum([len(doc['text']) for doc in docs])

for i, doc in enumerate(docs):
    words, lemmas = text_to_words(doc['text'])
    doc['words'] = words
    doc['lemmas'] = lemmas
    t_bytes = len(doc['text'])
    if round(c_bytes / n_bytes, 4) != round((c_bytes+t_bytes) / n_bytes, 4):
        print(str(round(100.0 * (c_bytes + t_bytes) / n_bytes, 2)) + r'% ' + time.strftime('%H:%M:%S'))
    c_bytes += t_bytes
# all_words = [lemma for doc in docs for lemma in doc['words']]
# unique_words = list(set(all_words))
# print(len(all_words), len(unique_words))

with open('text.pickle', 'wb') as pick:
    pickle.dump(docs, pick)