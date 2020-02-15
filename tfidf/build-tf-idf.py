#!/usr/bin/env python3
import pickle
from math import log

class TF_IDF(): # Sparse
    def __init__(self, docs):
        self.docs = docs
        self.terms = self.build_term_list()
        self.term_idx = {term : idx for idx, term in enumerate(self.terms)}
        self.n_docs = len(self.docs)
        self.n_terms = len(self.terms)
        self.build_bag_of_words()
        self.idf = self.build_idf()
        self.norm_bow()

    def build_term_list(self):
        all_words = [word for doc in self.docs for word in doc['words']]
        return list(set(all_words))

    def build_bag_of_words(self):
        for doc in self.docs:
            doc['bow'] = {}
            for word in doc['words']:
                if word in doc['bow']:
                    doc['bow'][word] += 1
                else:
                    doc['bow'][word] = 1

    def build_idf(self):
        doc_freq = {term : 0.0 for term in self.terms}
        for doc in self.docs:
            for word in doc['bow'].keys():
                doc_freq[word] += 1.0
        idf = {term : log((self.n_docs+1) / (doc_freq[term]+1)) + 1 for term in self.terms}
        return idf

    def build_tf_idf(self): # builds it into doc['bow']
        for doc in self.docs:
            for word in doc['bow'].keys():
                doc['bow'][word] *= self.idf[word]

    def norm_bow(self):
        for doc in self.docs:
            vlen = sum([freq * freq for freq in doc['bow'].values()]) ** 0.5
            for word in doc['bow'].keys():
                doc['bow'][word] /= vlen
    
    def save(self):
        self.docs = [{'title' : doc['title'], 'bow' : doc['bow'], 'len' : len(doc['words'])} for doc in self.docs]
        with open('tf_idf_min.pickle', 'wb') as pick:
            pickle.dump(self.docs, pick)
        with open('idf.pickle', 'wb') as pick:
            pickle.dump(self.idf, pick)

if __name__ == '__main__':

    with open('text-lower.pickle', 'rb') as pick:
        docs = pickle.load(pick)
        tf_idf = TF_IDF(docs)
        tf_idf.save()

# def build_dense_td_matrix(docs):
#     unique_terms, term_index = build_term_list(docs)

#     n_terms = len(unique_terms)
#     n_docs = len(docs)

#     for doc in docs:
#         for term in doc['bow'].keys():

#     for i, term in enumerate(unique_terms):
#         for doc in docs
#     td_matrix = np.zeros(shape=(n_docs, n_terms))
