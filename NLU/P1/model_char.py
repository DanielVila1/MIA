import csv
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from keras.utils import plot_model

from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation, Concatenate
from keras.optimizers import Adam

class charTagger(tf.keras.Model):
 
    def __init__(self, tokenizer_word, tokenizer_char, word_emb_dim=30, char_emb_dim=30, lstm_neurons=100, out_dim=17):
        super(charTagger, self).__init__()
        self.word_embd = tf.keras.layers.Embedding(len(tokenizer_word.word_index)+1, word_emb_dim, mask_zero=True, name='Word_Embedding_Layer')
        self.char_embd = tf.keras.layers.TimeDistributed(tf.keras.layers.Embedding(len(tokenizer_char.word_index)+1,char_emb_dim,mask_zero=True), name='Char_Embedding_Layer')
        self.char_lstm = tf.keras.layers.TimeDistributed(tf.keras.layers.LSTM(char_emb_dim,return_sequences=False, name='Char_LSTM_Layer'), name='Char_TimeDistributed_LSTM')
        self.lstm = tf.keras.layers.LSTM(lstm_neurons, return_sequences=True, name='LSTM_Layer')
        self.output_layer = tf.keras.layers.TimeDistributed(Dense(out_dim,activation="softmax"), name='Dense_TimeDistributed_Layer')

    def call(self, word_input_layer, char_input_layer):
        x=self.word_embd(word_input_layer)
        y=self.char_embd(char_input_layer)
        y=self.char_lstm(y)
        y=tf.keras.layers.concatenate([x,y], name='Concatenate_Layer')
        y=self.lstm(y)
        return self.output_layer(y)
        
    def build(self):
        word_input_layer = tf.keras.layers.Input(shape=(128), name='Word_Input_Layer')
        char_input_layer = tf.keras.layers.Input(shape=(128,15), name='Char_Input_Layer')
        return tf.keras.Model(inputs=[word_input_layer,char_input_layer], outputs=self.call(word_input_layer,char_input_layer))

    def show(self):
        return tf.keras.utils.plot_model(self.build(), show_shapes=True)

    def train(cls, train_inputs, train_char_inputs, train_targets, val_inputs, val_char_inputs, val_targets, loss='categorical_crossentropy', opt='adam', metrics=['accuracy'], epochs=10):
        model=cls.build()
        model.compile(loss=loss, optimizer=opt, metrics=metrics)
        model.fit(x=[train_inputs, train_char_inputs] , y= train_targets, validation_data=((val_inputs, val_char_inputs), val_targets), epochs=epochs)
        return model

    def evaluate(cls, test_inputs, test_char_inputs, test_targets, loss='categorical_crossentropy', opt='adam', metrics=['accuracy'], show=False):
        model=cls.build()
        model.compile(loss=loss, optimizer=opt, metrics=metrics)
        score=model.evaluate((test_inputs, test_char_inputs), test_targets)
        if show:
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
        return score

    def predict(cls, word_inputs, char_inputs):
        return  cls.build().predict([word_inputs,char_inputs])