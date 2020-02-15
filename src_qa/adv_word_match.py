from nltk.tokenize import word_tokenize
from .stemmer import stemmer
import math

number_special = "NumberUniqueWord"

question_dict = {
    "кога": ["когато", "през", "година", "месец", "януари", "февруари", "март", "април", "май", "юни", "юли", "август", "септември", "октомври", "ноември", "декември"],
    "колко": ["колкото", number_special],
    "къде": ["където", "в", "във", "върху"],
    "кой": [],
    "кои": [],
    "кое": [],
    "защо": [],
    "какво": [],
    "какъв": [],
    "какви": []
}

def string_to_word(s):
    if len(s)==0:
        return None
    if s[0] <= '9' and s[0] >= '0':
        return number_special
    if s.isalpha():
        return stemmer.stem(s)
        # return s
    return None

def tokenize(s):
    tokens = word_tokenize(s)
    ans = []
    for t in tokens:
        ans = ans + t.split('-')
    return ans

def get_words_array(s, question=False):
    ans = []
    tokens = tokenize(s)
    for i, word in enumerate(tokens):
        w = word.lower()
        if question and i==0 and w in question_dict:
            ans.append([string_to_word(x) for x in question_dict[w]])
        else:
            stemmed = string_to_word(w)
            if stemmed is not None and stemmed not in ans:
                ans.append(stemmed)
    return ans

def is_in(a, b):
    if isinstance(a, str):
        return a in b
    for x in a:
        if x in b:
            return True
    return False

def evaluate_word_match(question, sentences):
    question_words = get_words_array(question, True)
    freq = [0]*len(question_words)
    for sw in sentences:
        sent_words = get_words_array(sw[0])
        for i, w in enumerate(question_words):
            if is_in(w, sent_words):
                freq[i]+=1
    
    freq = [(1+math.log(len(sentences)/f) if f>0 else 0) for f in freq]
    freq[0]*=2
    
    # print(question_words)
    # print(freq)

    for sw in sentences:
        sent_words = get_words_array(sw[0])
        cnt = 0
        for i, w in enumerate(question_words):
            if is_in(w, sent_words):
                cnt+=freq[i]
        sw[1]*=cnt

