import math
import random
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from collections import defaultdict


class MarkovChain:
    def __init__(self, n):
        self.n = n
        self.tokenizer = None
        self.table = defaultdict(lambda: defaultdict(int))
    
    def fit_tokenizer(self, sentences):
        self.tokenizer = Tokenizer(oov_token=0)
        self.tokenizer.fit_on_texts(sentences)
        
    def fit(self, text):
        for sentence in text:
            tokens = self.tokenizer.texts_to_sequences([sentence])[0]

            for i in range(len(tokens)-self.n):
                context = tuple(tokens[i:i+self.n])
                next_word = tokens[i+self.n]
                self.table[context][next_word] += 1
             
                # print("Context: ", context)
                # print("Next word: ", next_word)
                # print("Occurence: ", self.table[context][next_word])
    
    def validate(self, text):
        perplexities = []
        vocab_size = len(self.tokenizer.word_index)
        # n_words = 0
        # for i in range(len(text)):
        #   n=len(txt[i])
        #   n_words += n

        for sentence in text:
            tokens = self.tokenizer.texts_to_sequences([sentence])[0]
            n = len(tokens)
            if n == 0:
                continue
            probs = []
            for i in range(n-self.n):
                context = tuple(tokens[i:i+self.n])
                next_word = tokens[i+self.n]
                prob = 1/((self.table[context][next_word] + 1) / (sum(self.table[context].values()) + vocab_size)) 
                probs.append(prob)
            prod_prob = np.prod(probs)
            #perplexity = np.power(prod_prob, 1/n_words) # Total words
            perplexity = np.power(prod_prob, 1/n) # Words of sentence
            if not np.isinf(perplexity):
              perplexities.append(perplexity)
        print(perplexities)
        return np.mean(perplexities)

  
    def generate_text(self, N, strategy='a'):
        context = random.choice(list(self.table.keys()))
        generated_text = list(context)
        context_txt = generated_text.copy()
        for _ in range(N):
            if context in self.table:
                word_probabilities = self.table[context]
                if strategy == 'a':
                    next_word = max(word_probabilities, key=word_probabilities.get)
                elif strategy == 'b':
                    words, probabilities = zip(*word_probabilities.items())
                    # print(words)
                    # print(probabilities)
                    if np.sum(probabilities) != 0:
                      next_word = random.choices(words, weights=probabilities)[0]
                    else:
                      break
                elif strategy == 'c':
                    sort_prob = sorted(word_probabilities.items(), key=lambda x: x[1], reverse=True)[:50]
                    words, probabilities = zip(*sort_prob)
                    # print(words)
                    # print(probabilities)
                    if np.sum(probabilities) != 0:
                      next_word = random.choices(words, weights=probabilities)[0]
                    else: 
                      break
                generated_text.append(next_word)
                context = tuple(generated_text[-self.n:])
            else:
                break
        return self.tokenizer.sequences_to_texts([context_txt])[0], ''.join(self.tokenizer.sequences_to_texts([generated_text])[0])