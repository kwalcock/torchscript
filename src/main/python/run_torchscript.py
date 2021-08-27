import os.path

import torch

from NerDatamodule import NerDatamodule
from ScriptedModel import ScriptedModel

from Shared import make_model, run_dataset

def main():
    datapath = "../../../../data/ner-conll/"
    embedpath = "../../../../data/glove.840B.300d.10f.txt"
    modelpath = "../../../../data/model.pt"
    datamodule = NerDatamodule(datapath, embedpath, batch_size = 1)
    example_crop = 50
    crop = 0

    datamodule.setup("training")
    print(f"Size of dataset: {len(datamodule)}")
    print(f"Size of vocab: {len(datamodule.train_dataset.vocab)}")

    def make_scripted_model(size_of_vocab, size_of_labels):
        # if the file exists, load it
        if os.path.isfile(modelpath):
            traced_forward = torch.jit.load(modelpath)
        else:
            model = make_model(size_of_vocab ,size_of_labels)
            example_input = torch.randint(1, 70000, (1, example_crop))
            traced_forward = torch.jit.trace(model, example_input)
            traced_forward.save(modelpath)
        # print(traced_forward.code)
        return ScriptedModel(traced_forward)

    model = make_scripted_model(len(datamodule.train_dataset.vocab))

    with torch.no_grad():
        run_dataset(model, "train", crop, datamodule.train_dataloader())
        run_dataset(model, "  val", crop, datamodule.val_dataloader())
        run_dataset(model, " test", crop, datamodule.test_dataloader())


if __name__ == "__main__":
    print(torch.__version__)   
    main()
