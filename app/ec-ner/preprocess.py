import random

from datasets import Dataset
from transformers import AutoTokenizer

from configuration import (
    GAIIC_FILE,
    GAIIC_TEST_FILE,
    LABELS,
    MAX_SENT_LEN,
    MODEL_NAME,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    VALID_RATIO,
)

LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def read_conll(path):
    sentences, cur = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if cur:
                    sentences.append(cur)
                    cur = []
                continue
            parts = line.split()  # 兼容空格 / Tab 分隔
            if len(parts) == 2:
                char, tag = parts[0], parts[1]
            else:  # 脏行兜底：整行当一个字，标 O
                char, tag = line[0], "O"
            prefix = tag.split("-")[0]  # 丢掉类型后缀
            if prefix not in LABEL2ID:
                prefix = "O"
            cur.append((char, prefix))
    if cur:
        sentences.append(cur)
    return sentences


def align_labels(word_ids, char_label_ids):
    labels, prev = [], None
    for wid in word_ids:
        if wid is None:
            labels.append(-100)
        elif wid != prev:
            labels.append(char_label_ids[wid])
        else:
            labels.append(-100)
        prev = wid
    return labels


def load_pool_and_test():
    pool, test = [], []
    if not GAIIC_FILE.exists():
        raise FileNotFoundError(f"缺少语料 {GAIIC_FILE.name}，请先运行 download_data.py")
    sents = read_conll(GAIIC_FILE)
    pool += [("gaiic", s) for s in sents]
    print(f"gaiic_train: {len(sents)} sentences -> pool")
    if GAIIC_TEST_FILE.exists() and GAIIC_TEST_FILE.stat().st_size >= 10_000:
        test = read_conll(GAIIC_TEST_FILE)
        print(f"gaiic_test: {len(test)} sentences -> test（官方测试集）")
    return pool, test


def _encode(sentences, tokenizer):
    examples = []
    for sent in sentences:
        chars = [c for c, _ in sent]
        inputs = tokenizer(chars, is_split_into_words=True, truncation=True)
        labels = align_labels(inputs.word_ids(), [LABEL2ID[t] for _, t in sent])
        examples.append(
            {
                "input_ids": inputs["input_ids"],
                "token_type_ids": inputs["token_type_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": labels,
            }
        )
    return examples


def process():
    pool, test = load_pool_and_test()
    kept = [(src, s) for src, s in pool if len(s) <= MAX_SENT_LEN]
    dropped = len(pool) - len(kept)
    if dropped:
        print(f"丢弃超长句(>{MAX_SENT_LEN}字): {dropped}")

    seen, uniq = set(), []
    for src, sent in kept:
        key = (tuple(c for c, _ in sent), tuple(t for _, t in sent))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(sent)
    print(f"训练池去重: {len(kept)} -> {len(uniq)} sentences")

    random.seed(RANDOM_SEED)
    random.shuffle(uniq)

    if test:
        n_valid = int(len(uniq) * VALID_RATIO)
        splits = {
            "train": uniq[n_valid:],
            "valid": uniq[:n_valid],
            "test": [s for s in test if len(s) <= MAX_SENT_LEN],
        }
        print(f"切分（官方 test + 训练集 9:{int(VALID_RATIO*100)}）: "
              f"train={len(splits['train'])} valid={len(splits['valid'])} "
              f"test={len(splits['test'])}")
    else:  # 对全量做 9:1:1 切分
        n = len(uniq)
        n_test = round(n / 11)        # 1/11
        n_valid = round(n / 11)       # 1/11
        n_train = n - n_test - n_valid  # 9/11
        splits = {
            "train": uniq[:n_train],
            "valid": uniq[n_train:n_train + n_valid],
            "test": uniq[n_train + n_valid:],
        }
        print(f"切分（全量 9:1:1）: "
              f"train={len(splits['train'])} valid={len(splits['valid'])} "
              f"test={len(splits['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, sentences in splits.items():
        ds = Dataset.from_list(_encode(sentences, tokenizer))
        ds.save_to_disk(PROCESSED_DATA_DIR / split)
        print(f"{split}: {len(ds)} sentences -> {PROCESSED_DATA_DIR / split}")


if __name__ == "__main__":
    process()