import torch

from NerDatamodule import NerDatamodule

from Shared import make_model, run_dataset

def main():
    datapath = "../../../../data/ner-conll/"
    embedpath = "../../../../data/glove.840B.300d.10f.txt"
    datamodule = NerDatamodule(datapath, embedpath, batch_size = 1)
    crop = 0

    print(f"Size of dataset: {len(datamodule)}")
    print(f"Size of vocab: {len(datamodule.train_dataset.vocab)}")

    model = make_model(len(datamodule.train_dataset.vocab), datamodule.num_classes)

    with torch.no_grad():
        run_dataset(model, "train", crop, datamodule.train_dataloader())
        run_dataset(model, "  val", crop, datamodule.val_dataloader())
        run_dataset(model, " test", crop, datamodule.test_dataloader())


if __name__ == "__main__":
    print(torch.__version__)
    main()
