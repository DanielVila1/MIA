import csv
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer

from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation, Concatenate
from keras.optimizers import Adam

class Parser:
    """
    Class defining the neural model. 
    """
        
    def __init__(self,n,emb_dim,tokenizer,tokenizer_relations):
        self.emb_dim=emb_dim
        self.n=n
        self.action_dim=4         
        self.tokenizer=tokenizer
        self.tokenizer_relations=tokenizer_relations
        self.action_dic=None
        self.model=None
        self.hist=None

        
    def build(self, summary=False):
        stack_input=tf.keras.layers.Input(shape=(self.n,),name='Stack_Input')
        buffer_input=tf.keras.layers.Input(shape=(self.n,),name='Buffer_Input')
        stack_emb=tf.keras.layers.Embedding(len(self.tokenizer.word_index)+1, self.emb_dim, mask_zero=True, name='Stack_Embedding')(stack_input)
        stack_emb=tf.keras.layers.Flatten()(stack_emb)
        buffer_emb=tf.keras.layers.Embedding(len(self.tokenizer.word_index)+1, self.emb_dim, mask_zero=True, name='Buffer_Embedding')(buffer_input)
        buffer_emb=tf.keras.layers.Flatten()(buffer_emb)

        concatenated_embeddings= tf.keras.layers.Concatenate(name='Concatenate')([stack_emb, buffer_emb])

        relation_output = tf.keras.layers.Dense(len(self.tokenizer_relations.word_index)+1, activation='softmax', name='Relation_Output')(concatenated_embeddings) 
        transition_output = tf.keras.layers.Dense(self.action_dim, activation='softmax', name='Transition_Output')(concatenated_embeddings)
        
        self.model=keras.Model(inputs = [stack_input, buffer_input], outputs = [transition_output, relation_output])
        if summary:
            self.model.summary()
    
    
    def compile(self, loss, optimizer, metrics):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        
        
    def show(self):
        return plot_model(self.model, show_shapes=True, show_layer_names=True)
        
        
    def train(self, train_stack,train_buffer,train_transitions,train_relations,val_stack,val_buffer,val_transitions,val_relations, epochs):
        self.hist = self.model.fit(x=[train_stack, train_buffer],y=[train_transitions, train_relations],validation_data=([val_stack,val_buffer],[val_transitions, val_relations]),epochs=epochs)

        
    def predict(self, stack, buffer):
        action, relation=self.model.predict([np.asarray(stack), np.asarray(buffer)], verbose=0)
        return action,relation
