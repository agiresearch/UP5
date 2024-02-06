from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import json
import torch
from transformers import T5Tokenizer
import argparse
from torch.utils.data.distributed import DistributedSampler
import random
import torch.distributed as dist

from data_preprocessing_movie import MovieTrainDataset


class InputDataset(Dataset):
    def __init__(self, args, data):
        self.args = args
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datapoint = self.data[index]
        input_text, output_text = datapoint

        return input_text, output_text


def calculate_whole_word_ids(tokenized_text, input_ids):
    whole_word_ids = []
    curr = 0
    for i in range(len(tokenized_text)):
        if tokenized_text[i] == "<pad>":
            curr = 0
        if tokenized_text[i].startswith("▁"):
            curr += 1
            whole_word_ids.append(curr)
        else:
            whole_word_ids.append(curr)
    return whole_word_ids[: len(input_ids) - 1] + [0]  # [0] for </s>


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_texts = [input_text[0] for input_text in batch]
        output_texts = [input_text[1] for input_text in batch]

        inputs = self.tokenizer.batch_encode_plus(input_texts, padding=True)
        input_ids = inputs["input_ids"]
        whole_word_ids = []
        for input_id in input_ids:
            tokenized_text = self.tokenizer.convert_ids_to_tokens(input_id)
            whole_word_id = calculate_whole_word_ids(tokenized_text, input_id)
            whole_word_ids.append(whole_word_id)
        input_attention = inputs["attention_mask"]
        outputs = self.tokenizer.batch_encode_plus(
            output_texts, padding="longest", truncation=True, max_length=512
        )
        output_ids = outputs["input_ids"]
        output_attention = outputs["attention_mask"]

        return (
            torch.tensor(input_ids),
            torch.tensor(input_attention),
            torch.tensor(whole_word_ids),
            torch.tensor(output_ids),
            torch.tensor(output_attention),
        )


def load_dataloaders(args, tokenizer):
    task = args.task
    data_dir = args.data_dir + task + "/"
    
    if args.task == 'movie':
        traindataset = MovieTrainDataset(args)
    else:
        train_dir = data_dir + task + "_train_data.json"
        with open(train_dir, "r") as f:
            train = json.load(f)
    print("loaded train")
        
    val_dir = data_dir + task + "_val_data.json"
    test_dir = data_dir + task + "_test_data.json"

    with open(val_dir, "r") as f:
        val = json.load(f)
    print("loaded val")

    with open(test_dir, "r") as f:
        test = json.load(f)

    collator = Collator(tokenizer)

    print("collator created")

    valdataset = InputDataset(args, val)
    testdataset = InputDataset(args, test)
    print("create dataset")

    if args.distributed:
        sampler = DistributedSampler(traindataset)
    else:
        sampler = None

    train_loader = DataLoader(
        traindataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        shuffle=(sampler is None),
        sampler=sampler,
    )
    print("create train loader")

    if args.distributed:
        sampler = DistributedSampler(valdataset)
    else:
        sampler = None
    val_loader = DataLoader(
        valdataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        shuffle=False,
        sampler=sampler,
    )
    print("create val loader")

    if args.distributed:
        sampler = DistributedSampler(testdataset)
    else:
        sampler = None
    test_loader = DataLoader(
        testdataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        shuffle=False,
        sampler=sampler,
    )
    print("create test loader")

    return train_loader, val_loader, test_loader



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/")

    parser.add_argument("--movie_category_negative_sample", type=int, default=5)
    parser.add_argument("--negative_sample", type=int, default=5)
    parser.add_argument("--sequential_num", type=int, default=25)
    parser.add_argument("--yes_no_sample", type=int, default=5)
    parser.add_argument("--max_history", type=int, default=20)

    parser.add_argument(
        "--task", type=str, default="movie", help="movie, insurance"
    )
    parser.add_argument("--toy", action="store_true")

    parser.add_argument("--batch_size", type=int, default=4)

    # data
    parser.add_argument("--use_item_representation", action="store_true")
    parser.add_argument(
        "--item_representation",
        type=str,
        default="random_vocab_mapping",
        help="hash_table, random_vocab_mapping, vocab_mapping",
    )
    parser.add_argument("--distributed", action="store_true")

    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    train_loader, val_loader, test_loader = load_dataloaders(args, tokenizer)

    for batch in train_loader:
        input_ids = batch[0][2]
        attn = batch[1][2]
        whole_words = batch[2][2]
        print(input_ids.tolist())
        print(attn.tolist())
        print(whole_words.tolist())

        print(tokenizer.convert_ids_to_tokens(input_ids))

        break
