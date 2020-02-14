import xml.etree.ElementTree as ET
import heapq
from simple import evaluate_simple
from adv_word_match import evaluate_word_match
from adv_emb_match import evaluate_emb_match

first_in_par_w = 1.2
in_first_par_w = 1.3

class QA:
    def __init__(self):
        self.evaluate_simple = evaluate_simple
        self.evaluate_word_match = evaluate_word_match
        self.evaluate_emb_match = evaluate_emb_match

    def innertext(self, tag):
        return (tag.text or '') + ''.join(self.innertext(e) for e in tag) + (tag.tail or '')

    def split_sentences(self, text):
        cap = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЪЮЯ"
        sent = ""
        last = ""
        ans = []
        for c in text:
            if not c.isspace() or c=='\n':
                if last == '.' or last == '\n' and c in cap:
                    ans.append(sent)
                    sent = ""
                last = c
            sent += c
        return ans

    def split_articles(self, articles, weights=None):
        ans = []
        for i in range(len(articles)):
            content = articles[i].find('content')
            for j in range(len(content)):
                txt = self.innertext(content[j])
                txts = self.split_sentences(txt)
                for k in range(len(txts)):
                    w=1
                    if weights is not None:
                        w*=weights[i]
                    if j==0:
                        w*=in_first_par_w
                    if k==0:
                        w*=first_in_par_w 
                    ans.append([txts[k], w])
        return ans

    def get_k_best(self, sentences, k):
        h = []
        for s in sentences:
            if len(h)<k:
                heapq.heappush(h, (s[1], s[0]))
            elif h[0][0] < s[1]:
                heapq.heapreplace(h, (s[1], s[0]))
        ans = [heapq.heappop(h) for i in range(len(h))]
        ans.reverse()
        ans = [(a[1], a[0]) for a in ans]
        return ans

    def query_simple(self, question, articles, weights=None, k=10):
        sentences = self.split_articles(articles, weights)
        self.evaluate_simple(question, sentences)
        return self.get_k_best(sentences, k)

    def query_adv_word_match(self, question, articles, weights=None, k=10):
        sentences = self.split_articles(articles, weights)
        self.evaluate_word_match(question, sentences)
        return self.get_k_best(sentences, k)
        
    def query_adv_emb_match(self, question, articles, weights=None, k=10):
        sentences = self.split_articles(articles, weights)
        self.evaluate_emb_match(question, sentences)
        return self.get_k_best(sentences, k)