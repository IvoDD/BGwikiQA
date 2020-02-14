from nltk.tokenize import word_tokenize

def get_words_array(s):
    return [word.lower() for word in word_tokenize(s) if word.isalpha()]

def evaluate_simple(question, sentences):
    q = get_words_array(question)
    for sw in sentences:
        s = get_words_array(sw[0])
        cnt = 0
        for token in q:
            if token in s:
                cnt += 1
        sw[1]*=cnt
    
