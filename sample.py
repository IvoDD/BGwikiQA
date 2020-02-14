from tfidf.query import query

docs = query('Машинно самообучение', 3)
print()
for doc in docs:
    print(doc['title'])
    print(doc['url'])
    print()