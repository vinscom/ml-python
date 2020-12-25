import csv
import unittest
from dataset.Dataset import DataSet
from dataset.GloVeDataset import GloveDataset
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import numpy as np
import os

class TextML:
    hp_embedding_dim = 100
    hp_max_length = 16
    hp_trunc_type = 'post'
    hp_padding_type = 'post'
    hp_oov_tok = '<OOV>'
    hp_training_size = 16000
    hp_test_portion = 0.1
    hp_epochs = 4

    def __init__(self):
        self.tokenizer = Tokenizer(oov_token='<OOV>')

    def train(self):
        # Tokenize Sentences
        dataset = DataSet('data/training_cleaned.csv')
        sentences, labels = dataset.training()
        test_sentences, test_labels = dataset.test()
        self.tokenizer.fit_on_texts(sentences)
        train_sentences = pad_sequences(self.tokenizer.texts_to_sequences(sentences), TextML.hp_max_length)
        train_labels = np.array(labels)
        val_sentences = pad_sequences(self.tokenizer.texts_to_sequences(test_sentences), TextML.hp_max_length)
        val_labels = np.array(test_labels)
        # Get Weights
        glove = GloveDataset('data/glove.6B.100d.txt')
        weights = glove.embedding_matrix(self.tokenizer.word_index)

        # Build Model
        input_dim = len(self.tokenizer.word_index) + 1
        emb_init = keras.initializers.Constant(weights)
        model = keras.Sequential([
            keras.layers.Embedding(input_dim, 100, input_length=TextML.hp_max_length, embeddings_initializer=emb_init,
                                   trainable=False),
            keras.layers.Bidirectional(keras.layers.LSTM(32)),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(train_sentences, train_labels, epochs=TextML.hp_epochs,
                            validation_data=(val_sentences, val_labels), verbose=2)
        model.save('newmode2.tf')
        print(history)
