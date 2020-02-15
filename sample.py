import xml.etree.ElementTree as ET
from tfidf.query import query
from src_qa.qa import QA

class AnswerQuestion:
    def __init__(self, corpus = 'bgwiki-20180920-corpus.xml'): # https://dumps.wikimedia.org/bgwiki/20200201/bgwiki-20200201-pages-articles.xml.bz2
        self.QA = QA()
        print('Loading corpus - start')
        self.root = ET.parse(corpus).getroot()
        print('Loading corpus - end')

    def query(self, question, qa_strategy=3, n_docs=10, n_answers=10):
        docs = query(question, n_docs)
        article_titles = [doc['title'] for doc in docs]
        articles = [self.root.find(f'./article[@name="{title}"]') for title in article_titles]
        articles = [art for art in articles if art is not None]
        if qa_strategy==1:
            answers = self.QA.query_simple(question, articles, None, n_answers)
        if qa_strategy==2:
            answers = self.QA.query_adv_word_match(question, articles, None, n_answers)
        if qa_strategy==3:
            answers = self.QA.query_adv_emb_match(question, articles, None, n_answers)
        return article_titles, [a[0] for a in answers]

if __name__ == '__main__':
    aq = AnswerQuestion()
    titles, answers = aq.query("Кога е започнала Втората Световна Война", 3)
    for t in titles:
        print(t)
    for a in answers:
        print(a)
# docs = query('Машинно самообучение', 3)
# print()
# for doc in docs:
#     print(doc['title'])
#     print(doc['url'])
#     print()