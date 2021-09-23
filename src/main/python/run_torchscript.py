import os.path

import torch
import torch.onnx

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

    print(f"Size of dataset: {len(datamodule)}")
    print(f"Size of vocab: {len(datamodule.train_dataset.vocab)}")

    def save_model(model, input):
        model.save(modelpath)
        print("The torchscript model was saved.")

    def make_scripted_model(size_of_vocab, size_of_labels):
        if os.path.isfile(modelpath):
            # If the file already exists, just load it.
            model = torch.jit.load(modelpath)
        else:
            example_input = torch.randint(1, size_of_vocab - 1, (1, example_crop))
            model = make_model(size_of_vocab, size_of_labels)
            traced_model = torch.jit.trace(model, example_input)
            save_model(traced_model, example_input)
        return ScriptedModel(model)

    model = make_scripted_model(len(datamodule.train_dataset.vocab), datamodule.num_classes)

    with torch.no_grad():
        run_dataset(model, "train", crop, datamodule.train_dataloader())
        run_dataset(model, "  val", crop, datamodule.val_dataloader())
        run_dataset(model, " test", crop, datamodule.test_dataloader())


if __name__ == "__main__":
    print(torch.__version__)   
    main()
