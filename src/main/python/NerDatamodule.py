import os

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from NerDataset import NerDataset


class NerDatamodule:
    def __init__(self, dataset_path, embedding_path, batch_size):
        super().__init__()
        self.dataset_path = dataset_path
        self.embed_path = embedding_path
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        self.train_dataset = NerDataset(
            os.path.join(self.dataset_path, "eng-2col.train"), self.embed_path
        )
        self.val_dataset = NerDataset(
            os.path.join(self.dataset_path, "eng-2col.testa"), self.embed_path
        )
        self.test_dataset = NerDataset(
            os.path.join(self.dataset_path, "eng-2col.testb"), self.embed_path
        )

    def setup(self, stage) -> None:
        self.size = (
            len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset)
        )
        self.num_classes = len(self.train_dataset.labels)

    def collate_fn(batch):
        out_tensors, out_labels = [], []
        for sent, labels in batch:
            out_tensors.append(sent)
            out_labels.append(labels)

        out_tensors = pad_sequence(out_tensors, batch_first=True, padding_value=0)
        out_labels = pad_sequence(out_labels, batch_first=True, padding_value=-1)
        return out_tensors, out_labels

    def __len__(self):
        return self.size

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=NerDatamodule.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=NerDatamodule.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=NerDatamodule.collate_fn,
        )
