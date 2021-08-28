import torch
from torch.utils.data import Dataset

from NerVocab import mask

class NerDataset(Dataset):
    def __init__(self, dataset_path, vocab):
        self.samples = []
        self.labels = {}
        self.vocab = vocab

        # Load data and construct label map
        with open(dataset_path, "r") as data_file:
            cur_sentence = [[], []]
            for line in data_file:
                cur = line.strip()
                if cur == "":
                    self.samples.append(cur_sentence)
                    cur_sentence = [[], []]
                else:
                    token, label = cur.split()
                    cur_sentence[0].append(token)
                    cur_sentence[1].append(label)

                    if label not in self.labels:
                        # We will sort and replace number later
                        self.labels[label] = -1

        for i, label in enumerate(sorted(self.labels.keys())):
            self.labels[label] = i

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        assert len(self.samples) > index and index >= 0
        sentence, labels = self.samples[index]

        out_sent = []
        for token in sentence:
            if token in self.vocab:
                out_sent.append(self.vocab[token])
            else:
                out_sent.append(self.vocab[mask])

        out_labels = [self.labels[label] for label in labels]

        out_sent = torch.tensor(out_sent)
        out_labels = torch.tensor(out_labels)

        return (out_sent, out_labels)
