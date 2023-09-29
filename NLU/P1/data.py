import csv
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.utils import pad_sequences, to_categorical
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

def flatten(l): [item for sublist in l for item in sublist]

class ProcessData:
    def __init__(self, dataset=None):
        self.dataset=dataset

    def create_inputs(self):
        inputs=[]
        aux_inputs = []
        for i in self.dataset:
            if i.startswith('#'):  
                continue
            if len(i) == 0:
                inputs.append(aux_inputs)
                aux_inputs = []
                continue
            aux0 = i.split("\t")[0]
            if ("-" in aux0) or ("." in aux0):  
                continue
            aux_ = i.split("\t")
            aux_inputs.append(aux_[1])
        return inputs

    def create_targets(self):
        targets=[]
        aux_targets = []
        for i in self.dataset:
            if i.startswith('#'):  
                continue
            if len(i) == 0:
                targets.append(aux_targets)
                aux_targets = []
                continue
            aux0 = i.split("\t")[0]
            if ("-" in aux0) or ("." in aux0):  
                continue
            aux_ = i.split("\t")
            aux_targets.append(aux_[3])
        return targets

    # def create_char(self):
    #     char_inputs=[]
    #     aux1=[]
    #     aux2=[]
    #     for sentence in self.create_inputs():
    #         for word in sentence:
    #             for char in word:
    #                 aux1.append(char)
    #             aux2.append(aux1)
    #             aux1=[]
    #         char_inputs.append(aux2)
    #         aux2=[]
    #     return char_inputs 

    def create_char(self):
        inp=self.create_inputs()
        aux1=[]
        char_inputs=[]
        for idx,i in enumerate(inp):
            for ind,j in enumerate(inp[idx]):
                aux1.append(list(inp[idx][ind]))
            char_inputs.append(aux1)
            aux1=[]
        return char_inputs

    def inputs_to_ids(self,tokenizer,maxlen):
        sequences= tokenizer.texts_to_sequences(self.create_inputs())  
        inputs_ids = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding="post") 
        return inputs_ids

    def targets_to_ids(self, encoder, maxlen):
        targets=self.create_targets()
        num_taggs=len(encoder.classes_)
         
        encoded_targets=[]
        for ind, i in enumerate(targets):
            encoded_targets.append(encoder.transform(targets[ind]))    

        targets_padded = tf.keras.preprocessing.sequence.pad_sequences(encoded_targets, maxlen=maxlen, padding="post")  
        targets_ids= tf.keras.utils.to_categorical(targets_padded, num_taggs)  
        return targets_ids

    # def char_to_ids(self, tokenizer_char, max_sentence_length, max_char_sequence_length):
    #     char=self.create_char()
    #     char_ids=[]
    #     aux=[]
    #     for idx, i in enumerate(char):
    #         y=tokenizer_char.texts_to_sequences(char[idx])
    #         y.extend([[0]] * ( max_sentence_length - len(y)))
    #         z=tf.keras.preprocessing.sequence.pad_sequences(y, max_char_sequence_length, padding="post")
    #         aux.append(z)
    #         char_ids.append(z)
    #         aux=[]
    #     return char_ids

    def char_to_ids(self, tokenizer_char, max_sentence_length, max_char_sequence_length):
        char_inputs=self.create_char()
        char_ids=[]
        for idx, char in enumerate(char_inputs):
            y = tokenizer_char.texts_to_sequences(char)
            char_padded = tf.keras.preprocessing.sequence.pad_sequences(y, max_char_sequence_length, padding="post")
            aux = [[0] * max_char_sequence_length]
            for i in range(max_sentence_length - len(char_padded)):
                char_padded=np.append(char_padded, aux, axis=0)
            char_ids.append(char_padded)
        return char_ids

    def char_inputs_adapted(self, tokenizer_char, max_sentence_length, max_char_sequence_length):
        char_inputs=self.char_to_ids(tokenizer_char, max_sentence_length, max_char_sequence_length)
        to_np = np.concatenate([np.array(i) for i in char_inputs])
        resized = np.resize(to_np, (len(char_inputs), max_sentence_length, max_char_sequence_length))
        return resized

    def decode(self, outputs, encoder):
        ouput_tags = []
        aux = []
        inputs=self.create_inputs()
        for idx, out1 in enumerate(outputs):
            l = len(inputs[idx])
            for idx, item in enumerate(out1):
                if idx > (l-1):
                    break
                id = np.where(item == np.amax(item))[0][0] 
                aux.append(id)
            ouput_tags.append(encoder.inverse_transform(aux))
            aux = []
        return ouput_tags