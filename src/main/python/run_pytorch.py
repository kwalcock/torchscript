import torch

from NerDatamodule import NerDatamodule
from LSTMModel import LSTMModel

from Shared import make_model

def main():
    datapath = "../../../../data/ner-conll/"
    embedpath = "../../../../data/glove.840B.300d.10f.txt"
    datamodule = NerDatamodule(datapath, embedpath, batch_size = 1)
    crop = 50

    datamodule.prepare_data()
    datamodule.setup("training")
    print(f"Size of dataset: {len(datamodule)}")
    print(f"Size of vocab: {len(datamodule.train_dataset.vocab)}")

    model = make_model(len(datamodule.train_dataset.vocab), datamodule.num_classes)

    with torch.no_grad():
        run_dataset("train", datamodule.train_dataloader())
        run_dataset("  val", datamodule.val_dataloader())
        run_dataset(" test", datamodule.test_dataloader())


if __name__ == "__main__":
    print(torch.__version__)
    main()
