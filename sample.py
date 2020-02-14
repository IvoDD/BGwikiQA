import xml.etree.ElementTree as ET
from tfidf.query import query
from src_qa.qa import QA

class AnswerQuestion:
    def __init__(self, corpus = 'bgwiki-20180920-corpus.xml'): # https://dumps.wikimedia.org/bgwiki/20200201/bgwiki-20200201-pages-articles.xml.bz2
        self.QA = QA()
        self.root = ET.parse(corpus).getroot()

    def query(self, question, n_docs=10, n_answers=10):
        docs = query(question, n_docs)
        article_ids = [doc['id'] for doc in docs]
        articles = [self.root.find(f"./article[@id={idx}]") for idx in article_ids]
        answers = self.QA.query_adv_emb_match(question, articles, None, n_answers)
        return [a[0] for a in answers]

aq = AnswerQuestion()
ans = aq.query("Кога е започнала Втората Световна Война")
for a in ans:
    print(a)
# docs = query('Машинно самообучение', 3)
# print()
# for doc in docs:
#     print(doc['title'])
#     print(doc['url'])
#     print()