import os.path

import torch
import torch.onnx

from NerDatamodule import NerDatamodule
from ScriptedModel import ScriptedModel

from Shared import make_model, run_dataset

def main():
    datapath = "../../../../data/ner-conll/"
    embedpath = "../../../../data/glove.840B.300d.10f.txt"
    modelpath = "../../../../data/model.onnx"
    datamodule = NerDatamodule(datapath, embedpath, batch_size = 1)
    example_crop = 50
    crop = 0

    print(f"Size of dataset: {len(datamodule)}")
    print(f"Size of vocab: {len(datamodule.train_dataset.vocab)}")

    def save_model(model, input):
        input_names = [ "input" ]
        output_names = [ "output" ]
        # Why does this issue a warning?
        torch.onnx.export(model, example_input, onnxpath, verbose = True, input_names = input_names, output_names = output_names)
        print("The onnx model was saved.")

    def make_scripted_model(size_of_vocab, size_of_labels):
        if False:
            # Don't bother to load it.
        else:
            example_input = torch.randint(1, size_of_vocab, (1, example_crop))
            model = make_model(size_of_vocab, size_of_labels)
            traced_model = torch.jit.trace(model, example_input)
            save_model(traced_model, example_input)
        return ScriptedModel(model)

    model = make_scripted_model(len(datamodule.train_dataset.vocab), datamodule.num_classes)

    with torch.no_grad():
        # Don't run the onnx model here
        # run_dataset(model, "train", crop, datamodule.train_dataloader())
        # run_dataset(model, "  val", crop, datamodule.val_dataloader())
        # run_dataset(model, " test", crop, datamodule.test_dataloader())


if __name__ == "__main__":
    print(torch.__version__)   
    main()
