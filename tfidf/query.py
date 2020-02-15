#!/usr/bin/env python3
import pickle
import stanfordnlp
import heapq
from math import log
import os

cdir = os.path.dirname(__file__)

nlp = stanfordnlp.Pipeline(processors='tokenize,pos', lang='bg')

idf = {}
docs = []

def init():
    global idf
    global docs
    print('Started initializing')
    with open(os.path.join(cdir, 'idf.pickle'), 'rb') as pick:
        idf = pickle.load(pick)

    with open(os.path.join(cdir, 'tf_idf_min.pickle'), 'rb') as pick:
        docs = pickle.load(pick)
    print('Done initializing')

def text_to_words(text):
    sentences = nlp(text).sentences
    tokens = [token for sentence in sentences for token in sentence.tokens]
    words = [word for token in tokens for word in token.words]
    words = [word for word in words if word.upos != 'PUNCT']
    # lemmas = [word.lemma for word in words]
    words = [word.text for word in words]
    return words

def query(text, k=5):
    words = [word.lower() for word in text_to_words(text)]
    terms = list(set(words))
    
    # Calculate term freq
    bow = {term : 0.0 for term in terms}
    for term in words:
        bow[term] += 1.0
    
    # Multiply by inverse doc freq
    for term in terms:
        bow[term] *= idf[term]
    
    # Normalize
    vlen = sum([freq * freq for freq in bow.values()]) ** 0.5
    for term in bow.keys():
        bow[term] /= vlen

    calc_dists(bow)

    k_best = heapq.nsmallest(k, docs, key=lambda doc: -doc['sim'])
    return k_best

def calc_dists(bow):
    print('Started calculating distances')
    for doc in docs:
        sim = 0.0
        for term in bow.keys():
            if term in doc['bow'].keys():
                sim += bow[term] * doc['bow'][term]
        doc['sim'] = sim * log(doc['len'])**2
    print('Done calculating distances')
    
init()
if __name__ == '__main__':
    while True:
        text = input('Enter query: ')
        print('Processing query')
        k_best = query(text)
        for doc in k_best:
            print(doc['title'])
