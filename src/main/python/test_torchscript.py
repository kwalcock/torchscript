import time
from statistics import stdev

import torch
from tqdm import tqdm

from NerDataloader import NerDatamodule
from LSTMModel import LSTMModel
from ScriptedModel import ScriptedModel


def main():
    datapath = "/data/nlp/corpora/processors/en/ner-conll/"
    embedpath = "/data/nlp/corpora/glove/glove.840B.300d.10f.txt"
    datamodule = NerDatamodule(datapath, embedpath, batch_size=1)

    datamodule.prepare_data()
    datamodule.setup("training")
    print(f"Size of dataset: {len(datamodule)}")
    print(f"Size of vocab: {len(datamodule.train_dataset.vocab)}")

    def make_model():
        model = LSTMModel(
            vocab_size=len(datamodule.train_dataset.vocab),
            embed_dim=300,
            label_size=datamodule.num_classes,
            hidden_dim=128,
        )
        example_input = torch.randint(1, 70000, (1, 50))
        traced_forward = torch.jit.trace(model, example_input)
        return ScriptedModel(traced_forward)

    model = make_model()

    def crop_input(sample):
        input_ids = sample[0]
        if input_ids.shape[1] >= 50:
            input_ids = input_ids[:, :50]
        else:
            new_inputs = torch.zeros((1, 50), dtype=torch.long)
            new_inputs[0, : input_ids.shape[1]] = input_ids[0, :]
            input_ids = new_inputs
        return input_ids

    with torch.no_grad():
        train_times = []
        for sample in tqdm(datamodule.train_dataloader()):
            cropped = crop_input(sample)
            start = time.time()
            model.forward(cropped)
            train_times.append(time.time() - start)
        print(f"Mean train sample time: {sum(train_times)/len(train_times)}")
        print(f"                 stdev: {stdev(train_times)}")

        val_times = []
        for sample in tqdm(datamodule.val_dataloader()):
            cropped = crop_input(sample)
            start = time.time()
            model.forward(cropped)
            val_times.append(time.time() - start)
        print(f"Mean val sample time: {sum(val_times)/len(val_times)}")
        print(f"                 stdev: {stdev(val_times)}")

        test_times = []
        for sample in tqdm(datamodule.test_dataloader()):
            cropped = crop_input(sample)
            start = time.time()
            model.forward(cropped)
            test_times.append(time.time() - start)
        print(f"Mean test sample time: {sum(test_times)/len(test_times)}")
        print(f"                 stdev: {stdev(test_times)}")


if __name__ == "__main__":
    main()
