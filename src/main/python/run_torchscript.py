import os.path
import time
from statistics import stdev

import torch
from tqdm import tqdm # A Fast, Extensible Progress Bar for Python and CLI

from NerDatamodule import NerDatamodule
from LSTMModel import LSTMModel
from ScriptedModel import ScriptedModel


def main():
    datapath = "../../../../data/ner-conll/"
    embedpath = "../../../../data/glove.840B.300d.10f.txt"
    modelpath = "../../../../data/model.pt"
    datamodule = NerDatamodule(datapath, embedpath, batch_size=1)
    example_crop = 50
    crop = 0

    datamodule.prepare_data()
    datamodule.setup("training")
    print(f"Size of dataset: {len(datamodule)}")
    print(f"Size of vocab: {len(datamodule.train_dataset.vocab)}")

    def make_model():
        # if the file exists, load it
        if os.path.isfile(modelpath):
            traced_forward = torch.jit.load(modelpath)
        else:
            model = LSTMModel(
                vocab_size=len(datamodule.train_dataset.vocab),
                embed_dim=300,
                label_size=datamodule.num_classes,
                hidden_dim=128,
            )
            example_input = torch.randint(1, 70000, (1, example_crop))
            traced_forward = torch.jit.trace(model, example_input)
            traced_forward.save(modelpath)
        # print(traced_forward.code)
        return ScriptedModel(traced_forward)

    model = make_model()

    def crop_input(sample):
        input_ids = sample[0]
        if crop <= 0 or input_ids.shape[1] == crop:
            pass
        else:
            if input_ids.shape[1] > crop:
                input_ids = input_ids[:, :crop]
            else:
                new_inputs = torch.zeros((1, crop), dtype=torch.long)
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

    def run_dataset(name, dataset):
        times = []
        for sample in dataset: # tqdm(dataset):
            cropped = crop_input(sample)
            start = time.time()
            result = model.forward(cropped)
            print_shape(result)
            print_data(result)
            times.append(time.time() - start)
        print(f"  Mean {name} sample time: {sum(times)/len(times):.8f}")
        print(f"Stddev {name} sample time: {stdev(times):.8f}")

    with torch.no_grad():
        run_dataset("train", datamodule.train_dataloader())
        run_dataset("  val", datamodule.val_dataloader())
        run_dataset(" test", datamodule.test_dataloader())


if __name__ == "__main__":
    print(torch.__version__)   
    main()
