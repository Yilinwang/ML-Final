from collections import Counter
import numpy as np
import gensim
import jieba
import os



class Sentences:
    def __init__(self, dir_path):
        self.counter = Counter()
        self.corpus = list()
        for path in os.listdir(dir_path):
            for line in open(dir_path + '/' + path):
                print(list(jieba.cut(line.strip())))
                input()
            #self.counter += Counter(open(dir_path + '/' + path).read())

        self.vocabulary = set([key for key, value in self.counter.items() if value > 5])
    
    def __iter__(self):
        for path in self.paths:
            for line in open(path):
                yield self.process_sentence(line)

    def process_sentence(self, line):
        try:
            words = line.strip().split(' +++$+++ ')[-1].split()
        except:
            words = line
        s = ['<s>']
        for w in words:
            if w in self.vocabulary:
                s.append(w)
            else:
                s.append('<unk>')
        s.append('</s>')
        return s


class Word2Vec:
    def __init__(self):
        self.train()

    def train(self):
        print("----Word2Vec model training----")
        self.sentences = Sentences(['data/training_label.txt', 'data/training_nolabel.txt', 'data/testing_data.txt'])
        self.model = gensim.models.Word2Vec(self.sentences)

    def getwv(self, sentence, max_len = 0):
        sentence = self.process_sentence(sentence)
        if max_len > 0 and max_len > len(sentence):
            sentence.extend(['</s>'] * (max_len - len(sentence)))
        return self.model.wv[sentence]

    def process_sentence(self, sentence):
        return self.sentences.process_sentence(sentence)
