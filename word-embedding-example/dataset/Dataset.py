import csv
import unittest
import random


class DataSet:
    def __init__(self, filepath, test_portion=0.1):
        self.filepath = filepath
        self.test_portion = test_portion
        self.corpus = []

    def load(self, shuffle=False, **kwargs):
        print('Loading Dataset File')
        if len(self.corpus) == 0:
            with open(self.filepath) as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    self.corpus.append([row[5], row[0]])

        if shuffle:
            random.shuffle(self.corpus)
        print('Loading Dataset Finished')
        return [s[0] for s in self.corpus], [0 if "0" == s[1] else 1 for s in self.corpus]

    def training(self, **kwargs):
        sentences, labels = self.load(**kwargs)
        trainlength = int(len(sentences) * (1 - self.test_portion))
        return sentences[:trainlength], labels[:trainlength]

    def test(self, **kwargs):
        sentences, labels = self.load(**kwargs)
        trainlength = int(len(sentences) * (1 - self.test_portion))
        return sentences[trainlength:], labels[trainlength:]


class DataSetTest(unittest.TestCase):
    def setUp(self):
        self.data = DataSet('data/test.csv')
        sentences, labels = self.data.load()
        self.sentences = sentences
        self.labels = labels

    def test_load_file(self):
        self.assertTrue(len(self.sentences) == 14)
        self.assertTrue(len(self.labels) == 14)
        self.assertTrue(self.sentences[0].find('You shoulda got David Carr of Third Day to do it') != -1)

    def test_train_data(self):
        s, _ = self.data.training()
        self.assertTrue(s[len(s) - 1].find('I just re-pierced my') != -1)

    def test_test_data(self):
        s, _ = self.data.test()
        self.assertTrue(s[0].find('And I thought the UA loss was embarrassing') != -1)


if __name__ == '__main__':
    unittest.main()
