import os

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from NerDataset import NerDataset
from NerVocab import read_vocab


class NerDatamodule:
    def __init__(self, dataset_path, embedding_path, batch_size):
        super().__init__()
        self.dataset_path = dataset_path
        self.embed_path = embedding_path
        self.batch_size = batch_size

        vocab = read_vocab(embedding_path)

        self.train_dataset = self.prepare_file("eng-2col.train", vocab)
        self.val_dataset = self.prepare_file("eng-2col.testa", vocab)
        self.test_dataset = self.prepare_file("eng-2col.testb", vocab)

        self.size = (
            len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset)
        )
        self.num_classes = len(self.train_dataset.labels)

    def prepare_file(self, filename, vocab):
        return NerDataset(
            os.path.join(self.dataset_path, filename), vocab
        )

    def collate_fn(batch):
        out_tensors, out_labels = [], []
        for sent, labels in batch:
            out_tensors.append(sent)
            out_labels.append(labels)

        out_tensors = pad_sequence(out_tensors, batch_first = True, padding_value = 0)
        out_labels = pad_sequence(out_labels, batch_first = True, padding_value = -1)
        return out_tensors, out_labels

    def __len__(self):
        return self.size

    def new_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size = self.batch_size,
            collate_fn = NerDatamodule.collate_fn,
        )

    def train_dataloader(self):
        return self.new_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self.new_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self.new_dataloader(self.test_dataset)
