import re
import codecs

class Stemmer:
    def __init__(self, rules_file = "stem_rules_context_1_1.txt"):
        self.rules = dict()
        with codecs.open(rules_file, encoding='utf-8') as stem_rules:
            for line in stem_rules:
                rule_match = re.match('([а-я]+) ==> ([а-я]+) ([0-9]+)', line)
                if rule_match:
                    self.rules[rule_match.group(1)] = rule_match.group(2)
    
    def stem(self, word):
        for i in range(len(word)):
            if word[i:] in self.rules:
                return word[:i] + self.rules[word[i:]]
        return word

stemmer = Stemmer()