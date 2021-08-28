import time
from statistics import stdev

import torch

from LSTMModel import LSTMModel
from tqdm import tqdm # A Fast, Extensible Progress Bar for Python and CLI

def make_model(size_of_vocab, size_of_labels):
    model = LSTMModel(
        vocab_size = size_of_vocab,
        embed_dim = 300,
        label_size = size_of_labels,
        hidden_dim = 128
    )
    return model

def crop_input(sample, crop):
    input_ids = sample[0]
    if crop <= 0 or input_ids.shape[1] == crop:
        pass
    else:
        if input_ids.shape[1] > crop:
            input_ids = input_ids[:, :crop]
        else:
            new_inputs = torch.zeros((1, crop), dtype = torch.long)
            new_inputs[:, :input_ids.shape[1]] = input_ids[:, :]
            input_ids = new_inputs
    return input_ids

def print_shape(tensor):
    shape = tensor.shape
    i = 0
    for value in shape:
        if i > 0:
            print(", ", end = "")
        print(value, end = "")
        i += 1
    print()

# https://stackabuse.com/python-how-to-flatten-list-of-lists/
def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

def print_data(tensor):
    data = tensor.data
    i = 0
    for value in flatten(data.tolist()):
        if i > 0:
            print(", ", end = "")
        print(f"{value:1.8f}", end = "")
        i += 1
    print()

def run_dataset(model, name, crop, dataset):
    times = []
    for sample in dataset: # tqdm(dataset):
        cropped = crop_input(sample, crop)
        start = time.time()
        result = model.forward(cropped)
        times.append(time.time() - start)
        # print_shape(result)
        # print_data(result)
    print(f"  Mean {name} sample time: {sum(times)/len(times):.8f}")
    print(f"Stddev {name} sample time: {stdev(times):.8f}")
