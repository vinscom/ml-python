import csv
import unittest
import random
import numpy as np


class GloveDataset:
    def __init__(self, filepath):
        self.filepath = filepath
        self.corpus = {}

    def load(self):
        print('Loading Weight files')
        if len(self.corpus) == 0:
            with open(self.filepath) as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    self.corpus[word] = coefs
        print('Finished Weight files')
        return self.corpus

    def embedding_matrix(self, word_index):
        corpus = self.load()
        vocab_size = len(word_index)
        emb_dim = len(next(iter(corpus.values())))
        embedding_matrix = np.zeros((vocab_size + 1, emb_dim))
        for word, i in word_index.items():
            try:
                emb_vector = corpus[word]
                if emb_vector is not None:
                    embedding_matrix[i] = emb_vector
            except KeyError:
                pass

        return embedding_matrix


class GloveDatasetTest(unittest.TestCase):
    def setUp(self):
        self.data = GloveDataset('data/glove.txt')
        self.weights = self.data.load()

    def test_file_parsing(self):
        self.assertTrue(len(self.weights) == 10)
        self.assertTrue(len(self.weights['and']) == 100)

    def test_embedding_matrix(self):
        word_idx = {
            'and': 1,
            'the': 2
        }
        matrix = self.data.embedding_matrix(word_idx)
        self.assertTrue(matrix.shape == (3, 100))


if __name__ == '__main__':
    unittest.main()
