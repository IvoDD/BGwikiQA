from nltk.tokenize import word_tokenize
from flair.data import Sentence
from flair.embeddings import WordEmbeddings
import numpy as np
import torch
from .hungarian import hungarian

str_to_embed = WordEmbeddings('bg-wiki')

number_special = "число"

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
        return s
    return None

def word_to_embed(w):
    s = Sentence(w)
    str_to_embed.embed(s)
    return s[0].embedding.to("cpu")

def tokenize(s):
    tokens = word_tokenize(s)
    ans = []
    for t in tokens:
        ans = ans + t.split('-')
    return ans

def get_emb_array(s, question=False):
    tokens = tokenize(s)
    ans = []
    for i, token in enumerate(tokens):
        w = string_to_word(token.lower())
        if w is None:
            continue
        if question and i==0 and w in question_dict:
            ans.append([word_to_embed(x) for x in question_dict[w]])
        else:
            ans.append(word_to_embed(w))
    return ans

def dist(e1, e2):
    return np.sqrt(torch.sum((e1-e2)**2).numpy())

def dist_or(e1, e2):
    if isinstance(e1, list):
        return min([dist(x, e2) for x in e1], default=0)
    return dist(e1, e2)

def eval_dist(q_embed, s_embed):
    while len(s_embed) < len(q_embed):
        s_embed.append(torch.zeros(300))
    matr = np.zeros((len(q_embed), len(s_embed)))
    for i in range(len(q_embed)):
        for j in range(len(s_embed)):
            matr[i, j] = dist_or(q_embed[i], s_embed[j])
    
    return hungarian(matr)

def evaluate_emb_match(question, sentences):
    question_embed = get_emb_array(question, True)

    for sw in sentences:
        sent_embed = get_emb_array(sw[0])
        dist = eval_dist(question_embed, sent_embed)
        sw[1] = -dist/sw[1]

