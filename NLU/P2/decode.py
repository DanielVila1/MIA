import numpy as np
from oracle import *

def decode(predictions, action_dic, tokenizer_relations):
    """"" 
    Function to decode the predictions of the model.
    
    """"" 
    action_predicted=list(action_dic.keys())[list(action_dic.values()).index(np.argmax(predictions[0]))]
    relation_predicted=tokenizer_relations.sequences_to_texts([[np.argmax(predictions[1])]])[0]

    return action_predicted,relation_predicted


def prepare_sample(sent,s):
    """"" 
    Function to prepare the sample to feed the model with the current state and predict the next action.
    
    """"" 
    # state= [stack,buffer,A]
    # A= (head, label, child)
    if len(s[0]) < 2:
        top_stack=['EMPTY', sent[s[0][-1]]]
    else:
        top_stack=[sent[s[0][-1]], sent[s[0][-2]]]
    if len(s[1]) < 2:
        top_buffer=[sent[s[1][0]],'EMPTY']
    else:
        top_buffer=[sent[s[1][0]], sent[s[1][1]]]
    return top_stack, top_buffer


def is_valid_action(s):
    """"" 
    Function to check if the action predicted is valid.
    
    """"" 
    stack = s[0]
    buffer = s[1]
    relations = s[2]

    reduce = larc = rarc = 0

    # Reduce
    reduce1 = True
    if (len(stack) == 0): # No se si hay que comprobar los 'Empty'
        reduce1 = False

    if len(s[2])!=0:
        reduce2 = False
        for r in relations:
            if r[2] == stack[-1]: 
                reduce2 = True

        if reduce1 and reduce2:
            reduce = 1
    else: 
        reduce = 0

    # Right Arc
    if len(s[2])!=0:
        rarc1 = True
        for r in relations:
            if r[2] == buffer[0]: 
                rarc1 = False

        if rarc1:
            rarc = 1
    else:
        rarc=1

    # Left arc
    larc1 = True
    if stack[-1] == 0:
        larc1 = False
    else:
        if len(s[2])!=0:
            larc2 = True
            for r in relations:
                if r[2] == stack[-1]: 
                    larc2 = False

            if larc1 and larc2:
                larc = 1
        else:
            larc = 1


    # Shift

    shift = 1

    return [larc, rarc, shift, reduce]


# def predict(sents, my_parser, tokenizer, action_dic, tokenizer_relations):
#     """"" 
#     Function to predict the arcs of the sentences.
    
#     """"" 
    
#     arcs=[]
#     for sent in sents:
#         s=create_initial_state(sent)
#         while not is_final_state(s):
#             ts,tb=prepare_sample(sent,s) #top stack and buffer
#             stack=tokenizer.texts_to_sequences([ts])
#             buffer=tokenizer.texts_to_sequences([tb])
        
#             res=my_parser.predict(stack,buffer)
    
#             action,label=decode(res,action_dic,tokenizer_relations)
#             x=np.multiply(res[0],is_valid_action(s))
#             action =list(action_dic.keys())[list(action_dic.values()).index(np.argmax(x[0]))]

#             s=apply_action(s,action,label)   
#         arcs.append(s[2])     
#     return arcs


def predict(sents, my_parser, tokenizer, action_dic, tokenizer_relations):
    """"" 
    Function to predict the arcs of the sentences.
    
    """"" 
    arcs=[]
    for sent in sents:
        s=create_initial_state(sent)
        while not is_final_state(s):
            ts,tb=prepare_sample(sent,s) #top stack and buffer
            stack=tokenizer.texts_to_sequences([ts])
            buffer=tokenizer.texts_to_sequences([tb])
        
            res=my_parser.predict(stack, buffer)
           
            action,label=decode(res,action_dic,tokenizer_relations)
            x=np.multiply(res[0],is_valid_action(s))
            action =list(action_dic.keys())[list(action_dic.values()).index(np.argmax(x[0]))]

            s=apply_action(s,action,label)  
             
        # Avoid multiple roots as head
        bool_aux = False
        arcs_new = []
        root_word = 0
        for arc in s[2]:

            if arc[0] == 0:
                if bool_aux:
                    #arc_aux = [root_word, arc[1], arc[2]]
                    arc_aux = [root_word, 'nmod', arc[2]]
                else:
                    bool_aux = True
                    root_word = arc[2]
                    arc_aux = arc
            else:
                arc_aux = arc

            arcs_new.append(arc_aux)
        arcs.append(arcs_new)     
    return arcs


def conll_tree(sentence, sentence_arcs):
	
    conll_lines = []
    root_as_head = 0
    root_appeared = False
    for idx,word in enumerate(sentence):
        head_ = -1
        if word == "ROOT":
            continue
        for sentence_arc in sentence_arcs:
            if sentence_arc[2] == idx:
                if sentence_arc[0] == 0:
                    head_ = 0
                    relation_ = "root"
                    root_as_head = idx
                    root_appeared = True
                else:
                    head_ = sentence_arc[0]
                    relation_ = sentence_arc[1]
            """
            if sentence_arc[2] == idx and sentence_arc[0] == 0:
                if word == "BuzzMachine":
                    print("entra")
                head_ = 0
                relation_ = "root"
                root_as_head = idx
                root_appeared = True
            elif idx == sentence_arc[2]:
                head_ = sentence_arc[0]
                relation_ = sentence_arc[1]"""

        # No relation
        if head_ == -1:
             if root_appeared:
                 
                 head_ = root_as_head
                 relation_ = "dep"
             else:
                 root_appeared = True
                 root_as_head = idx
                 head_ = 0
                 relation_ = "root"
        if word == "BuzzMachine":
            print(head_)
            print(sentence_arc[0])
        conll_lines.append([idx, word, head_, relation_])
    return conll_lines


def create_trees_from_predicted_arcs(test, predicted_arcs):
    """"" 
    Function to create the trees from the predicted arcs.
    
    """"" 
    trees = []
    for test_i, arcs_i in zip(test, predicted_arcs):
        trees.append(conll_tree(test_i, arcs_i))
    return trees
        
        
def save_predicted_arcs(predicted_arcs):
    """"" 
    Function to save the predicted arcs on a file.
    
    """"" 
    with open('arcs_aux.txt', 'w') as output_file:
        for arc in predicted_arcs:
            output_file.write(str(arc) + "\n")


def str2tupleList(s):
    return eval( "[%s]" % s )

def load_predicted_arcs():
    # Load the arcs
    f = open('arcs_aux.txt', 'r') 
    arcs=[]
    for x in f:
        arcs.append(str2tupleList(x)[0])
    return arcs
            
def save_trees_conllu(trees):
    """"" 
    Function to save the trees on a file in conllu format.
    
    """"" 
    with open('output.conllu', 'w') as output_file:
        for t in trees:
            for sent in t:
                str_aux = ""
                
                """for col in sent:
                    str_aux = str_aux + (str(col)+ "\t")"""
                str_aux = str(sent[0]) + "\t" + str(sent[1]) + "\t" + "_" + "\t" + "_" + "\t" + "_" + "\t" + "_" + "\t" + str(sent[2]) + "\t" + str(sent[3]) + "\t" + "_" + "\t" + "_"

                output_file.write(str_aux + "\n")
            output_file.write("\n")
                
            
def save_test_data(test_data):
    with open('gold.conllu', 'w', encoding="utf-8") as output_file:
        output_file.write(test_data)
