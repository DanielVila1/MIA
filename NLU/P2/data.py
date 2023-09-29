import csv
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

import sys
from oracle import *


class ProcessData:
    def __init__(self, dataset=None):
        self.dataset=dataset


    def create_data(self):
        """""
        Function to obtain the inputs and the targets of the dataset excluding the lines that are not useful for the task.
        This way we obtain the words, the words ids, the UPOS tags the head ids and the dependency labels.
        
        """""
        inputs = []
        aux_inputs = [['0', 'ROOT', 'ROOT', '', '']]
        d=[]
        # we obtain the inputs and the targets -- words and its tags
        for i in self.dataset:
            if i.startswith('#'):  # we are not interested in this lines of the data
                continue
            if len(i) == 0:
                inputs.append(aux_inputs)
                aux_inputs = [['0', 'ROOT', 'ROOT', '', '']]
                continue
            aux0 = i.split("\t")[0]
            if ("-" in aux0) or ("." in aux0):  # we ignore the lines with the contractions of the language
                continue
            aux_ = i.split("\t")
            d.append(aux_[0]) # word id
            d.append(aux_[1]) # word
            d.append(aux_[3]) # UPOS
            d.append(aux_[6]) # head id
            d.append(aux_[7]) # dependency label
            aux_inputs.append(d)
            d=[]
        return inputs
    
    
    def preprocess_tree(self):
        """""
        Function to obtain the words ids and the head ids of the data in order to create the tree structure.
        
        """""      
        tree_inputs = []
        aux_inputs = []
        d=[]
        # we obtain the inputs and the targets -- words and its tags
        for i in self.dataset:
            if i.startswith('#'):  # we are not interested in this lines of the data
                continue
            if len(i) == 0:
                tree_inputs.append(aux_inputs)
                aux_inputs = []
                continue
            aux0 = i.split("\t")[0]
            if ("-" in aux0) or ("." in aux0):  # we ignore the lines with the contractions of the language
                continue
            aux_ = i.split("\t")
            d.append(aux_[0])
            d.append(aux_[6])
            aux_inputs.append(d)
            d=[]
        return tree_inputs
    
    
    def filter_projective_data(self):
        """""
        Function to filter the data that is non-projective.
        
        """"" 
        inputs=self.create_data()
        tree_inputs=self.preprocess_tree()
        filtered_inputs = []
        # filtered_tree_inputs = []
        for idx, sentence in enumerate(tree_inputs):
            if is_projective(sentence):
                # filtered_tree_inputs.append(sentence)
                filtered_inputs.append(inputs[idx])
        return filtered_inputs
    
    
    def get_wordsandPoS(self):
        """""
        Function to extract the words and the PoS tags of the dataset.
        
        """"" 
        inputs=self.filter_projective_data()
        res=[]
        aux=[]
        for sentence in inputs:
            d=[]
            for word in sentence:
                d.append(word[1])
                d.append(word[2])
                aux.append(d)
                d=[]
            res.append(aux)
            aux=[]
        return res
    
    
    def get_tokenizer_inputs(self):
        """"" 
        Function to create the inputs for the tokenizers and the encoders.
        
        """"" 
        inputs=self.create_data()
        tokenizer_inputs = []
        tokenizerUPOS_inputs = []
        tokenizerDEPREL_inputs = []
        for i in inputs:
            for y in i:
                tokenizer_inputs.append(y[1])
                tokenizerUPOS_inputs.append(y[2])
                tokenizerDEPREL_inputs.append(y[4])
            
        return tokenizer_inputs, tokenizerUPOS_inputs, tokenizerDEPREL_inputs
    
    
    def preprocess_oracle(self):
        """"" 
        Function to preprocess the data for the oracle model.
        
        """"" 
        oracle_inputs = []
        aux_inputs = [['0', '', '']]
        d=[]
        # we obtain the inputs and the targets -- words and its tags
        for i in self.dataset:
            if i.startswith('#'):  # we are not interested in this lines of the data
                continue
            if len(i) == 0:
                oracle_inputs.append(aux_inputs)
                aux_inputs = [['0', '', '']]
                continue
            aux0 = i.split("\t")[0]
            if ("-" in aux0) or ("." in aux0):  # we ignore the lines with the contractions of the language
                continue
            aux_ = i.split("\t")
            d.append(aux_[0])
            d.append(aux_[6])
            d.append(aux_[7])
            aux_inputs.append(d)
            d=[]
        return oracle_inputs
    
    
    def filter_projective_oracle(self):
        """"" 
        Function to filter the data that is non-projective for the oracle model.
        
        """"" 
        tree_inputs=self.preprocess_tree()
        oracle_inputs=self.preprocess_oracle()
        filtered_oracle=[]
        for idx, sentence in enumerate(tree_inputs):
            if is_projective(sentence):
                filtered_oracle.append(oracle_inputs[idx]) 
        return filtered_oracle
    

    def get_dependency_parse(self):
        """"" 
        Function to create the dependency parse of the sentence.
        
        """"" 
        outputs=[]
        for sentence in self.filter_projective_oracle():
            t=parse(sentence)
            outputs.append(t[2])
        return outputs
    
    
    def get_model_data(self, tokenizer, action_dic, tokenizer_relations):
        """"" 
        Function to create the data to feed the neural network.
        
        """"" 
        stack_inputs = []
        buffer_inputs = []
        action_outputs = []
        relation_outputs = []
        
        oracle_inputs=self.filter_projective_oracle()
        inputs=self.get_wordsandPoS()

        for idx in range(len(oracle_inputs)):
            stack_input, buffer_input, actions_t, relations_t=parse_train(oracle_inputs[idx],inputs[idx])

            stack_inputs.append(tokenizer.texts_to_sequences(stack_input))
            buffer_inputs.append(tokenizer.texts_to_sequences(buffer_input))
            action_outputs.append([action_dic.get(t) for t in actions_t])
            relation_outputs.append(tokenizer_relations.texts_to_sequences(relations_t))
            
        words_stack=[]
        for i in stack_inputs:
            for j in i:
                words_stack.append(j)
                
        words_buffer=[]
        for i in buffer_inputs:
            for j in i:
                words_buffer.append(j)
                
        actions_out=[]
        for i in action_outputs:
            for j in i:
                actions_out.append(j)
                
        relations_out=[]
        for i in relation_outputs:
            for j in i:
                relations_out.append(j)
        
        return words_stack, words_buffer, actions_out, relations_out
    
    
    def prepare_model_data(self, tokenizer, action_dic, tokenizer_relations):
        """""
        Function to pass output data to categorical and inputs to numpy arrays.
        
        """""    
        
        words_stack,words_buffer,actions_out,relations_out=self.get_model_data(tokenizer, action_dic, tokenizer_relations)    
        actions_ids2= tf.keras.utils.to_categorical(actions_out, 4) 
        relations_ids2= tf.keras.utils.to_categorical(relations_out, len(tokenizer_relations.word_index)+1)
        words_stack2=np.asarray(words_stack)
        words_buffer2=np.asarray(words_buffer)
        
        return  words_stack2, words_buffer2, actions_ids2, relations_ids2
    
    
    def create_test_data(self):
        inputs = []
        aux_inputs = ['ROOT']
        d=[]
        # we obtain the inputs and the targets -- words and its tags
        for i in self.dataset:
            if i.startswith('#'):  # we are not interested in this lines of the data
                continue
            if len(i) == 0:
                inputs.append(aux_inputs)
                aux_inputs = [ 'ROOT']
                continue
            aux0 = i.split("\t")[0]
            if ("-" in aux0) or ("." in aux0):  # we ignore the lines with the contractions of the language
                continue
            aux_ = i.split("\t")
            d.append(aux_[1]) # word
            aux_inputs.append(aux_[1])
            d=[]
        return inputs  
    

def is_projective(arcs: list):
    """
    Determines if a dependency tree has crossing arcs or not.
    Parameters:
    arcs (list): A list of tuples of the form (headid, dependentid, coding
    the arcs of the sentence, e.g, [(0,3), (1,4), …]
    Returns:
    A boolean: True if the tree is projective, False otherwise
    """
    for (i,j) in arcs:
        for (k,l) in arcs:
            if (i,j) != (k,l) and min(i,j) < min(k,l) < max(i,j) < max(k,l):
                return False
    return True
    
