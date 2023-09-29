import csv
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from keras.utils import plot_model

from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation, Concatenate
from keras.optimizers import Adam

class wordTagger(tf.keras.Model):
 
    def __init__(self, tokenizer, emb_dim=60, lstm_neurons=256, output_dim=17):
        super(wordTagger, self).__init__()
        self.embedding_layer=tf.keras.layers.Embedding(len(tokenizer.word_index) +1, emb_dim, mask_zero = True, input_length=128, name='Embedding_Layer')
        self.lstm_layer=tf.keras.layers.LSTM(lstm_neurons, return_sequences=True, name='LSTM_Layer')
        self.output_layer=tf.keras.layers.TimeDistributed(Dense(output_dim, activation="softmax"), name='TimeDistributed_Dense_Layer')

    def call(self, inputs,  **kwargs):
        x=self.embedding_layer(inputs)
        x=self.lstm_layer(x)
        return self.output_layer(x)
        
    def build(self):
        x = tf.keras.layers.Input(shape=(128, ), name='Input_Layer')
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def show(self):
        return tf.keras.utils.plot_model(self.build(), show_shapes=True)

    def train(cls, train_inputs, train_targets, val_inputs, val_targets, loss='categorical_crossentropy', opt='adam', metrics=['accuracy'], epochs=10):
        model=cls.build()
        model.compile(loss=loss, optimizer=opt, metrics=metrics)
        train_ds = tf.data.Dataset.from_tensor_slices((train_inputs,train_targets))
        train_ds = train_ds.batch(64)  
        #callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]
        model.fit(train_ds, validation_data=(np.array(val_inputs), val_targets), epochs=epochs)
        return model

    def evaluate(cls, test_inputs, test_targets, loss='categorical_crossentropy', opt='adam', metrics=['accuracy'], show=False):
        model=cls.build()
        model.compile(loss=loss, optimizer=opt, metrics=metrics)
        score=model.evaluate(test_inputs, test_targets)
        if show:
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
        return score

    def predict(cls, inputs):
        return  cls.build().predict(inputs)