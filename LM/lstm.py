from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import numpy as np
import pandas as pd
import random

class TextGenerator:
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.vocab_size = len(self.tokenizer.word_index)
        self.model = None
        self.hist = None
        
    def generate_sequences_ids(self, sentences):
        contexts_ids=[]
        labels_ids=[]
        for sentence in sentences:
            words = sentence.split()
            word_ids = self.tokenizer.texts_to_sequences([words])[0]
            for i in range(len(words)):
                context_ids = word_ids[max(0, i - self.max_seq_length):i]
                label_id = word_ids[i]
                contexts_ids.append(context_ids)
                labels_ids.append(label_id)
        return contexts_ids, labels_ids
    
    def generate_sequences(self, sentences):
        # Generate context and label pairs for each sentence
        contexts, labels = self.generate_sequences_ids(sentences)
        # Pad the sequences
        contexts_padded = pad_sequences(contexts, maxlen=self.max_seq_length, padding='pre')
        # One hot encode the labels
        labels_one_hot = to_categorical(labels, num_classes= self.vocab_size + 1)
        return contexts_padded, labels_one_hot
    
    def generate_sequences_words(self, sentences):
        contexts=[]
        for sentence in sentences:
            words = sentence.split()
            for i in range(len(words)):
                context = words[max(0, i - self.max_seq_length):i+1]
                contexts.append(context)
        return contexts
        
    def build(self, embedding_size, lstm_size, summary=False):
        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size+1, embedding_size, input_length=self.max_seq_length))
        self.model.add(Bidirectional(LSTM(lstm_size, return_sequences=True)))
        self.model.add(LSTM(lstm_size))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(self.vocab_size+1, activation='softmax'))
        if summary:
            self.model.summary()
    
    def compile(self, loss, optimizer, metrics):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics, run_eagerly=True)
        
    def train_model(self, contexts, labels, contexts_val, labels_val, epochs=20, batch_size=128):
        self.hist = self.model.fit(contexts, labels, epochs=epochs, batch_size=batch_size, validation_data=(contexts_val, labels_val))
    
    def calculate_perplexity(self, test_sentences):
        perplexities = []
        for sentence in test_sentences:
            ctxs, _ = self.generate_sequences([sentence])
            n = len(ctxs)
            if n == 0: 
                continue
            sentence_prob = 1
            preds = self.model.predict(ctxs, verbose = 0)[0]
            for i in range(n):
                sentence_prob *= 1 / preds[i]
            perplexity = np.power(sentence_prob, 1/n)
            if not np.isinf(perplexity):
              perplexities.append(perplexity)
        print(perplexities)
        return np.mean(perplexities)
    
    def generate_context(self, sentences):
        contexts = self.generate_sequences_words(sentences)
        rdm = random.choice(contexts)
        ctx = ' '.join(rdm)
        return ctx 
     
    def generate_text(self, context, N, strategy='a'):
        context_tokens = self.tokenizer.texts_to_sequences([context])[0]
        generated = []
        for _ in range(N):
            # Pad the sequence to the maximum length
            padded_context = pad_sequences([context_tokens], maxlen=self.max_seq_length)
            # Make the prediction using the LSTM model
            preds = self.model.predict(padded_context, verbose = 0)[0]
            if strategy == 'a':
                # Get the index of the most likely next word
                next_word_index = np.argmax(preds)
                if next_word_index==1:
                    sorted_indices = np.argsort(preds)
                    next_word_index = sorted_indices[-2]
            elif strategy == 'b':
                candidate_indices = [i for i in range(len(preds)) if i != 1]
                candidate_probs = preds[candidate_indices]
                candidate_probs /= candidate_probs.sum()
                next_word_index = np.random.choice(candidate_indices, 1, p=candidate_probs)[0]
            elif strategy == 'c':
                top_50_prob = np.argsort(preds)[-50:]
                top_50_prob = [i for i in top_50_prob if i != 1]  # remove <UNK> token
                top_50_pred = [preds[i] for i in top_50_prob]
                top_50_pred_normalized = [float(i)/sum(top_50_pred) for i in top_50_pred]
                next_word_index = np.random.choice(top_50_prob, 1, p=top_50_pred_normalized)[0]

            generated.append(next_word_index)
            context_tokens.append(next_word_index)
            context_tokens=context_tokens[1:] 
            generated_words = [self.tokenizer.index_word[word_index] for word_index in generated]
        return context + ' ' + ' '.join(generated_words) + '.'