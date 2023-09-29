import numpy as np

"""""
Functions to create and execute the oracle for the dependency parser task.

"""""

def create_initial_state(sentence):
    """""
    Funtion to create the initial state of the oracle from a given sentence.
    
    """""
    # [σ, β, A] = [[ROOT], [w1, w2, ...]., {}]

    alpha = [0]
    betha = list(range(1,len(sentence)))
    A = []
    return [alpha, betha, A]
    
    
def LA_is_valid(state):
    """""
    Function to check if the LA action is valid in a given state of the oracle.
      
    """""    
    # σ|i
    i = state[0][-1]
    #j|β
    j = state[1][0]
    # ¬[i = 0]
    if i == 0:
        return False
    # ¬∃k∃l [(k, l , i) ∈ A]
    for s in state[2]:
        if s[2] == i:
            return False
    return True
        
        
def RA_is_valid(state):
    """""
    Function to check if the RA action is valid in a given state of the oracle.
      
    """""
    # σ|i
    i = state[0][-1]
    #j|β
    j = state[1][0]
    # ¬∃k∃l [(k, l , j) ∈ A]
    for s in state[2]:
        if s[2] == j:
            return False
    return True


def REDUCE_is_valid(state):
    """""
    Function to check if the REDUCE action is valid in a given state of the oracle.
        
    """""
    # σ|i
    i = state[0][-1]
    # ∃k∃l[(k, l, i) ∈ A]
    for s in state[2]:
        if s[2] == i:
            return True
    return False
    
    
def apply_LA(state,sentence):
    """""
    Function to apply the LA action in a given state of the oracle.
    
    """""    
    # (σ|i, j|β, A) ⇒ (σ, j|β, A∪{(j, l, i)})
    # A element ()
    i=state[0][-1]
    j=state[1][0]
    # Add a dependency arc (j, l, i) to A, where i is the node on top of the 
    # stack σ and j is the first node in the buffer β
    state[2].append((state[1][0], sentence[i][2], state[0][-1]))
    # pop the stack σ
    state[0].pop(-1)
    return state


def apply_LA_(state,label):
    """""
    Function to apply the LA action in a given state of the oracle.
    
    """""    
    # Add a dependency arc (j, l, i) to A, where i is the node on top of the 
    # stack σ and j is the first node in the buffer β
    state[2].append((state[1][0], label, state[0][-1]))
    # pop the stack σ
    state[0].pop(-1)
    return state


def apply_RA(state,sentence):
    """""
    Funtion to apply the RA action in a given state of the oracle.
    
    """""    
    # (σ|i, j|β, A) ⇒ (σ|i|j,β, A∪{(i, l, j)})
    # *!* l ????
    i=state[0][-1]
    j = state[1][0]
    # Add a dependency arc (i, l, j) to A, where i is the node on top of the 
    # stack σ and j is the first node in the buffer β
    state[2].append((state[0][-1], sentence[j][2], state[1][0]))
    # remove the first node j in the buffer β and push it on top of the stack σ
    state[0].append(j)
    state[1].pop(0)
    return state

def apply_RA_(state,label):
    """""
    Funtion to apply the RA action in a given state of the oracle.
    
    """""    
    # (σ|i, j|β, A) ⇒ (σ|i|j,β, A∪{(i, l, j)})
    # *!* l ????
    j = state[1][0]
    # Add a dependency arc (i, l, j) to A, where i is the node on top of the 
    # stack σ and j is the first node in the buffer β
    state[2].append((state[0][-1], label, state[1][0]))
    # remove the first node j in the buffer β and push it on top of the stack σ
    state[0].append(j)
    state[1].pop(0)
    return state


def apply_REDUCE(state):
    """""
    Function to apply the REDUCE action in a given state of the oracle.
    """""
    # (σ|i,β, A) ⇒ (σ,β, A)
    # pops the stack 
    state[0].pop(-1)
    return state


def apply_SHIFT(state):
    """""
    Function to apply the SHIFT action in a given state of the oracle.
    
    """""
    # remove the first node i in the buffer β and pushes it on top of the stack σ
    i=state[1][0]
    state[1].pop(0)
    state[0].append(i)
    return state


def is_final_state(state):
    """""
    Function to check if the state is the final state of the oracle.
    
    """""
    if state[1]==[]:
        return True
    else:
        return False
    
    
def has_all_children(stack_top,state, sentence):
    """""
    Function to check if the stack_top has all its children in the state.
    
    """""
    children_sentence = sum([str(stack_top) in t for t in sentence])
    children_state = sum([(stack_top) in t for t in state[2]])
    if children_sentence==children_state:
        return True
    else:
        return False
    
    
def apply_action(s, action, label):
    if action=='LA':
        return apply_LA_(s,label)
    elif action=='RA':
        return apply_RA_(s,label)
    elif action=='REDUCE':
        return apply_REDUCE(s)
    else:
        return apply_SHIFT(s)
    
    
def oracle(state,sentence):
    """""
    Function to determinate the next action of the parsing.
    
    """""
	#   s,b = top_word_from_stack, top_word_from_buffer
    s = state[0][-1] 
    b = state[1][0]  
    
    if LA_is_valid(state) and int(sentence[s][1])==b:
        return 'LA'
    elif RA_is_valid(state) and int(sentence[b][1])==s:
        return 'RA'
    elif REDUCE_is_valid(state) and has_all_children(s,state,sentence):
        return 'REDUCE'
    else :
        return 'SHIFT'


def parse(sentence):
    """""
    Function to parse a sentence using the oracle algorithm and obtain the target dependency tree.
    
    """""
	# sentence ["word id", "head id", "dependency label"]
    state = create_initial_state(sentence) 
    while not is_final_state(state):
        t=oracle(state,sentence)
		# Left Arc
        if t=="LA":
            state=apply_LA(state,sentence)
		# Right arc
        elif t=="RA":
            state=apply_RA(state,sentence)
		#Reduce
        elif t=="REDUCE":
            state=apply_REDUCE(state)
		#SHIFT
        else:
            state=apply_SHIFT(state)
    return state 


def parse_train(sentence,input):
    """""
    Function to parse a sentence using the oracle algorithm and obtain the target dependency tree 
    and the actions and relations in tuples to feed the neural network.
    
    Sentence: given in the oracle structure
    Input: original sentence inputs (id, word, pos, head, relation)
    
    """""    
    state = create_initial_state(sentence) 
    stack_inputs=[]
    buffer_inputs=[]
    actions_t=[]
    relations_t=[]
    
    while not is_final_state(state):
        relation_aux = False
        t=oracle(state,sentence)
        if len(state[0]) < 2:
            top_s = ['EMPTY', input[state[0][-1]][0]]
        else:
            top_s = [input[state[0][-2]][0], input[state[0][-1]][0]]
        if len(state[1]) < 2:
            top_b=[input[state[1][0]][0], 'EMPTY']
        else:
            top_b=[input[state[1][0]][0], input[state[1][1]][0]]
            
        if t=="LA":
            relation_aux = True
            state=apply_LA(state,sentence)
        elif t=="RA":
            relation_aux = True
            state=apply_RA(state,sentence)
        elif t=="REDUCE":
            state=apply_REDUCE(state)
        else:
            state=apply_SHIFT(state)
            
        if relation_aux:
            relation = state[2][-1][1]
        else:
            relation = 'None'
            
        stack_inputs.append(top_s)
        buffer_inputs.append(top_b)
        actions_t.append(t)
        relations_t.append(relation)
    
    return stack_inputs, buffer_inputs, actions_t, relations_t         

            