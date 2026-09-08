import re
from collections import Counter

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
UNK_IDX = 1

def tokenize(text: str) -> list[str]:
    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+",
        "<url>",
        text
    )

    text = re.sub(
        r"@\w+",
        "<user>",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text.split()

def build_vocab(texts, min_freq: int = 5, max_vocab_size: int | None = None):
    counter = Counter()

    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)

    word_idx = {
        PAD_TOKEN: PAD_IDX,
        UNK_TOKEN: UNK_IDX
    }

    tokens = counter.most_common()

    for token, frequency in tokens:
        if frequency < min_freq:
            continue

        if max_vocab_size is not None:
            if len(word_idx) >= max_vocab_size:
                break

            word_idx[token] = len(word_idx)

    idx_word = {
        idx: word
        for word, idx in word_idx.items()
    }
    
    return word_idx, idx_word