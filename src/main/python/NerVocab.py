mask = "<MASK>"

def read_vocab(embedding_path):
    vocab = {}

    # Load word embeddings, (we just care about which tokens are in our vocab)
    with open(embedding_path, "r") as embed_file:
        lineno = 0
        for line in embed_file:
            if lineno > 0:
                cur = line[:line.index(" ")]
                vocab[cur] = -1
            lineno += 1

    # Add one so that we can preserve index 0 for our padding
    for i, token in enumerate(sorted(vocab.keys())):
        vocab[token] = i + 1
        # print(f"{i + 1}\t{token}")
    vocab[mask] = 0
    # print(f"{vocab[mask]}\t{mask}")

    return vocab
