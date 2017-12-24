from keras.preprocessing.sequence import pad_sequences
from collections import Counter
import numpy as np
import pickle
import jieba
import os



class Data:
    def __init__(self, dir_path):
        self.counter = Counter()
        self.corpus = list()
        for path in os.listdir(dir_path):
            for line in open(dir_path + path):
                self.counter += Counter(list(line.strip()))
                self.corpus.append(line.strip())
        self.vocabulary = {char: idx + 3 for idx, char in enumerate([key for key, value in self.counter.items() if value > 1])}
        self.training_data = pad_sequences([[1] + [self.vocabulary.get(char, 2) for char in line] + [0] for line in self.corpus], padding = 'post')
        self.maxlen = len(self.training_data[0])

    def get_training_data(self):
        return self.training_data

    def process_sentence(self, sentences):
        return pad_sequences([[1] + [self.vocabulary.get(char, 2) for char in sentence] + [0] for sentence in sentences], padding = 'post', maxlen = self.maxlen)

    def get_vocab_size(self):
        return len(self.vocabulary) + 3


if __name__ == '__main__':
    read_train_data('data/training_data')
