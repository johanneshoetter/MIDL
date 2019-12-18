from collections import defaultdict
import numpy as np
from sklearn.utils import shuffle
from copy import deepcopy
import operator

def build_map_classes_buffer(inputs, targets):
    '''
    Build a dictionary which has the targets as keys and all the corresponding inputs as values
    -- Params:
    @inputs: input data for the model
    @targets: desired outcome for the model given that it gets @inputs
    -- Return: dictionary
    '''
    # build a dictionary to grab pairs from
    map_classes_buffer = defaultdict(list) # stores class -> inputs pairs
    list(map(lambda x, y: map_classes_buffer[y].append(x), inputs, targets)) #without list, the writes to the map are not commited
    return map_classes_buffer

def sort_one_class(inputs, targets, batch_size, use_shuffle=True, random_state=None):
    '''
    Splits the data (inputs and targets) into as many homogeneous batches as possible,
    i.e. that in as many cases as possible, the batches consist of one target only.
    -- Params:
    @inputs: input data for the model
    @targets: desired outcome for the model given that it gets @inputs
    @batch_size: number of samples per batch
    @use_shuffle: shuffle the data beforehand
    -- Return: 2d list
    '''
    
    assert len(inputs) == len(targets), 'Inputs and targets do not have the same size'
    
    if use_shuffle:        
        inputs, targets = shuffle(inputs, targets, random_state=random_state)
    
    input_batches = []
    target_batches = []
    inputs_buffer = [] # store inputs that don't fit into homogeneous batches anymore
    targets_buffer = []
    
    num_batches = np.ceil(len(inputs) / batch_size)
    map_classes_buffer = build_map_classes_buffer(inputs, targets)
    
    for target in map_classes_buffer.keys():
        while len(map_classes_buffer[target]) > 0:
            taken_inputs, retrieved_inputs = map_classes_buffer[target][:batch_size], map_classes_buffer[target][batch_size:]
            if len(taken_inputs) < batch_size:
                inputs_buffer.extend(taken_inputs)
                targets_buffer.extend([target for taken_input in taken_inputs])
            else:
                input_batches.extend(taken_inputs)
                target_batches.extend([target for taken_input in taken_inputs])
            map_classes_buffer[target] = retrieved_inputs
            
    # take missing values
    while len(inputs_buffer) > 0:
        taken_inputs, inputs_buffer = inputs_buffer[:batch_size], inputs_buffer[batch_size:]
        taken_targets, targets_buffer = targets_buffer[:batch_size], targets_buffer[batch_size:]
        input_batches.extend(taken_inputs)
        target_batches.extend(taken_targets)
                
    return np.array(input_batches), np.array(target_batches) 

def sort_all_classes(inputs, targets, batch_size, use_shuffle=True, random_state=None):
    '''
    Splits the data (inputs and targets) into as many purely heterogeneous batches as possible,
    i.e. that in as many cases as possible, the batches consist of all targets.
    -- Params:
    @inputs: input data for the model
    @targets: desired outcome for the model given that it gets @inputs
    @batch_size: number of samples per batch
    @use_shuffle: shuffle the data beforehand
    -- Return: 2d list
    '''
    
    def try_pop(buffer):
        try:
            return buffer.pop()
        except:
            pass
            
    
    assert len(inputs) == len(targets), 'Inputs and targets do not have the same size'
    
    if use_shuffle:
        inputs, targets = shuffle(inputs, targets, random_state=random_state)
    
    input_batches = []
    target_batches = []
    inputs_buffer = [] # store inputs that don't fit into purely heterogeneous batches anymore
    targets_buffer = []
    
    num_batches = np.ceil(len(inputs) / batch_size)
    map_classes_buffer = build_map_classes_buffer(inputs, targets)
    
    # check if all targets in map class buffer have at least one element
    sorted_inputs = []
    sorted_targets = []
    copy_map_classes_buffer = deepcopy(map_classes_buffer)
    while any(list(map(lambda x: len(map_classes_buffer[x]) > 0, map_classes_buffer))):
        taken_targets = list(map(lambda x: x if try_pop(copy_map_classes_buffer[x]) is not None else None,\
                                 copy_map_classes_buffer)) # get one 'column'
        taken_inputs = list(map(lambda x: try_pop(map_classes_buffer[x]), map_classes_buffer)) # get one 'column'
        sorted_inputs.extend(taken_inputs)
        sorted_targets.extend(taken_targets)
        
    sorted_inputs = [val for val in sorted_inputs if val is not None] # None vals due to try_pop workaround
    sorted_targets = [val for val in sorted_targets if val is not None]
    while len(sorted_inputs) > batch_size:
        input_batches.extend(sorted_inputs[:batch_size])
        target_batches.extend(sorted_targets[:batch_size])
        sorted_inputs, sorted_targets = sorted_inputs[batch_size:], sorted_targets[batch_size:]
        
    if len(sorted_inputs) > 0: # if there are any values left
        input_batches.extend(sorted_inputs)
        target_batches.extend(sorted_targets)
                
    return np.array(input_batches), np.array(target_batches) 

def weighted_highest_sampling(inputs, targets, weighted_indices, batch_size=64):
    '''
    In weighted random sampling (WRS) the items are weighted and the probability of 
    each item to be selected is determined by its relative weight.
    -- Params:
    @inputs: input data for the model
    @targets: desired outcome for the model given that it gets @inputs
    @weighted_indices: dictionary containing which index of the samples has which probability
    @batch_size: number of samples per batch
    -- Return: one batch of the inputs, one batch of the targets
    '''
    
    assert len(inputs) == len(targets), 'The number of inputs to pull from must fit the number of targets'
    assert len(inputs) == len(weighted_indices), 'The number of inputs to pull from must fit the given weighted indices'
    assert type(weighted_indices) == dict, 'The weighted indices must be given as a dictionary'
      
    pulled_samples = []
    pulled_targets = []
    for _ in range(batch_size):
        pulled_index = max(weighted_indices.items(), key=operator.itemgetter(1))[0]
        weighted_indices[pulled_index] = 0
        pulled_samples.append(inputs[pulled_index])
        pulled_targets.append(targets[pulled_index])
    return pulled_samples, pulled_targets