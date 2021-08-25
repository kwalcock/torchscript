import torch
from torch.utils.data import Dataset


class NerDataset(Dataset):
    def __init__(self, dataset_path, embedding_path):
        self.samples = []
        self.labels = {}
        self.vocab = {}

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

        # Load word embeddings, (we just care about which tokens are in our vocab)
        with open(embedding_path, "r") as embed_file:
            for line in embed_file:
                cur = line.strip().split()[
                    0
                ]  # Format is: token e_0 e_1 ... e_n-2 e_n-1
                self.vocab[cur] = -1

        # Add one so that we can preserve index 0 for our padding
        for i, token in enumerate(sorted(self.vocab.keys())):
            self.vocab[token] = i + 1
        self.vocab["<MASK>"] = len(self.vocab)

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
                out_sent.append(self.vocab["<MASK>"])

        out_labels = [self.labels[label] for label in labels]

        out_sent = torch.tensor(out_sent)
        out_labels = torch.tensor(out_labels)

        return (out_sent, out_labels)
