import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import csv
import time
import numpy as np

from transformers import T5Tokenizer
from dataset import calculate_whole_word_ids
from torch.utils.data.distributed import DistributedSampler
from data_preprocessing_movie import MovieTrainDataset


def data_balance(args, data, user_feature):
    balanced_data = {}
    for datapoint in data:
        input_sentence = datapoint[0]
        assert "User_" in input_sentence or "user_" in input_sentence
        # for movie dataset, we only care about user information: gender, age
        user_id = "0"
        for word in input_sentence.split(" "):
            if word.startswith("User_"):
                user_id = word.replace("User_", "")
                break
            if word.startswith("user_"):
                user_id = word.replace("user_", "")
                break
        assert user_id != "0"
        if "gender" in args.feature:
            gender_info = {"F": 0, "M": 1}
            feature = user_feature[user_id]["gender"]
            feature = gender_info[feature]
            if feature in balanced_data.keys():
                balanced_data[feature].append(datapoint)
            else:
                balanced_data[feature] = [datapoint]
        elif "marital" in args.feature:
            marital_info = {
                "U": 0,
                "S": 1,
                "M": 2,
            }
            feature = user_feature[user_id]["marital"]
            feature = marital_info[feature]
            if feature in balanced_data.keys():
                balanced_data[feature].append(datapoint)
            else:
                balanced_data[feature] = [datapoint]
        elif "occupation" in args.feature:
            occupation_info = {
                "T4MS": 0,
                "90QI": 1,
                "56SI": 2,
            }
            feature = user_feature[user_id]["occupation"]
            feature = occupation_info[feature]
            if feature in balanced_data.keys():
                balanced_data[feature].append(datapoint)
            else:
                balanced_data[feature] = [datapoint]
        elif "age" in args.feature:
            feature = user_feature[user_id]["age"]
            feature = age_label(feature)
            if feature in balanced_data.keys():
                balanced_data[feature].append(datapoint)
            else:
                balanced_data[feature] = [datapoint]

    numbers = [len(d) for d in balanced_data.values()]
    mean = int(np.median(numbers))
    data = []
    for d in balanced_data.values():
        if len(d) > mean:
            data += random.sample(d, k=mean)
        else:
            data += d
    print(len(data))

    return data


def load_dataloaders(args):
    train, test, user_feature = load_data(args)

    train_dataset = IDInput(args, train, user_feature)
    test_dataset = IDInput(args, test, user_feature)

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    collator = Collator(tokenizer)

    if args.distributed:
        sampler = DistributedSampler(train_dataset)
    else:
        sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.discriminator_batch_size,
        collate_fn=collator,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
    )

    if args.distributed:
        sampler = DistributedSampler(test_dataset)
    else:
        sampler = None
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.discriminator_batch_size,
        collate_fn=collator,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
    )

    return train_loader, test_loader


def load_insurance_dataloaders(args):
    train, test, user_feature = load_insurance_data(args)

    train = data_balance(args, train, user_feature)

    train_dataset = InsuranceIDInput(args, train, user_feature)
    test_dataset = InsuranceIDInput(args, test, user_feature)

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    collator = Collator(tokenizer)

    if args.distributed:
        sampler = DistributedSampler(train_dataset)
    else:
        sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.discriminator_batch_size,
        collate_fn=collator,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
    )

    if args.distributed:
        sampler = DistributedSampler(test_dataset)
    else:
        sampler = None
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.discriminator_batch_size,
        collate_fn=collator,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
    )

    return train_loader, test_loader


def load_data(args):
    task = args.task

    if args.train_initial_discriminator:
        if "gender" in args.feature:
            with open(args.selected_train_data, "r") as f:
                train = json.load(f)
            with open(args.selected_test_data, "r") as f:
                test = json.load(f)
        elif "occupation" in args.feature:
            with open(args.occupation_train_data, "r") as f:
                train = json.load(f)
            with open(args.occupation_test_data, "r") as f:
                test = json.load(f)
        else:
            with open(args.train_data, "r") as f:
                train = json.load(f)
            with open(args.test_data, "r") as f:
                test = json.load(f)

    else:
        with open(args.train_data, "r") as f:
            train = json.load(f)

        with open(args.test_data, "r") as f:
            test = json.load(f)

    if args.discriminator_training:
        train = random.sample(train, k=50000)
        test = random.sample(test, k=10000)

    random.shuffle(train)

    if args.toy:
        train = train[:10]
        test = test[:10]

    # user data
    data_dir = args.data_dir + task + "/"
    with open(data_dir + "users.dat", "r", encoding="ISO-8859-1") as f:
        user_data = f.read()
    user_feature = {}
    user_data = user_data.split("\n")
    user_data.remove("")
    for d in user_data:
        d = d.split("::")
        index = d[0]
        gender = d[1]
        age = d[2]
        occupation = d[3]
        user_feature[index] = {"gender": gender, "age": age, "occupation": occupation}

    return train, test, user_feature


def load_insurance_data(args):
    task = args.task
    task_type = args.insurance_type

    if task_type == "sequential":
        if "occupation" in args.feature:
            with open(args.insurance_occupation_train_data, "r") as f:
                train = json.load(f)
            with open(args.insurance_occupation_test_data, "r") as f:
                test = json.load(f)
        elif "marital" in args.feature:
            with open(args.insurance_marital_train_data, "r") as f:
                train = json.load(f)
            with open(args.insurance_marital_test_data, "r") as f:
                test = json.load(f)
        else:
            with open(args.insurance_train_data, "r") as f:
                train = json.load(f)
            with open(args.insurance_test_data, "r") as f:
                test = json.load(f)
    elif task_type == "direct":
        if "occupation" in args.feature:
            with open(args.direct_insurance_occupation_train_data, "r") as f:
                train = json.load(f)
            with open(args.direct_insurance_occupation_test_data, "r") as f:
                test = json.load(f)
        elif "marital" in args.feature:
            with open(args.direct_insurance_marital_train_data, "r") as f:
                train = json.load(f)
            with open(args.direct_insurance_marital_test_data, "r") as f:
                test = json.load(f)
        else:
            with open(args.direct_insurance_train_data, "r") as f:
                train = json.load(f)
            with open(args.direct_insurance_test_data, "r") as f:
                test = json.load(f)

    random.shuffle(train)

    if args.toy:
        train = train[:100]
        test = test[:100]

    # user data
    data_dir = args.data_dir + task + "/"
    user_info = []
    with open(data_dir + "Train.csv", newline="\n") as f:
        reader = csv.reader(f, delimiter=",")
        for l in reader:
            user_info.append(l)
    user_feature = {}
    for d in user_info:
        user_id = d[0]
        gender = d[2]
        marital = d[3]
        occupation = d[7]
        age = d[4]
        user_feature[user_id] = {
            "gender": gender,
            "marital": marital,
            "occupation": occupation,
            "age": age,
        }

    return train, test, user_feature


def compute_keyword_ids(input_ids, tokenizer, keyword):
    indices = []
    for input_id in input_ids:
        tokens = tokenizer.convert_ids_to_tokens(input_id)
        assert "▁" + keyword in tokens
        if "▁" + "User" in tokens:
            start = tokens.index("▁User")
        else:
            start = tokens.index("▁" + keyword)
        end = start
        for i in range(start + 1, len(tokens)):
            if tokens[i].startswith("▁"):
                end = i
                break
        assert end != start
        indices.append([start, end])
    return indices


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_texts = [input_text[0] for input_text in batch]
        output_texts = [input_text[1] for input_text in batch]
        label = [input_text[2] for input_text in batch]

        inputs = self.tokenizer.batch_encode_plus(input_texts, padding=True)
        input_ids = inputs["input_ids"]
        user_ids = compute_keyword_ids(input_ids, self.tokenizer, "user")
        whole_word_ids = []
        for input_id in input_ids:
            tokenized_text = self.tokenizer.convert_ids_to_tokens(input_id)
            whole_word_id = calculate_whole_word_ids(tokenized_text, input_id)
            whole_word_ids.append(whole_word_id)
        input_attention = inputs["attention_mask"]

        outputs = self.tokenizer.batch_encode_plus(output_texts, padding=True)
        output_ids = outputs["input_ids"]
        output_attention = outputs["attention_mask"]

        return (
            torch.tensor(input_ids),
            torch.tensor(input_attention),
            torch.tensor(whole_word_ids),
            torch.tensor(output_ids),
            torch.tensor(output_attention),
            torch.tensor(user_ids),
            torch.tensor(label),
        )


class IDInput(Dataset):
    def __init__(self, args, data, user_feature):
        self.args = args
        self.data = data
        self.user_feature = user_feature

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datapoint = self.data[index]
        input_sentence = datapoint[0]
        target_sentence = datapoint[1]
        assert "user_" in input_sentence
        # for movie dataset, we only care about user information: gender, age
        if self.args.feature.startswith("user"):
            user_id = "0"
            for word in input_sentence.split(" "):
                if word.startswith("user_"):
                    user_id = word.replace("user_", "")
                    break
            assert user_id != "0"
            if "gender" in self.args.feature:
                gender_info = {"F": 0, "M": 1}
                feature = self.user_feature[user_id]["gender"]
                feature = gender_info[feature]
            elif "age" in self.args.feature:
                age_info = {
                    "1": 0,
                    "18": 1,
                    "25": 2,
                    "35": 3,
                    "45": 4,
                    "50": 5,
                    "56": 6,
                }
                feature = self.user_feature[user_id]["age"]
                feature = age_info[feature]
            else:
                feature = int(self.user_feature[user_id]["occupation"])
                # feature = occupation_label(self.user_feature[user_id]["occupation"])

        return input_sentence, target_sentence, feature


class InsuranceIDInput(Dataset):
    def __init__(self, args, data, user_feature):
        self.args = args
        self.data = data
        self.user_feature = user_feature

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datapoint = self.data[index]
        input_sentence = datapoint[0]
        target_sentence = datapoint[1]
        assert "User_" in input_sentence or "user_" in input_sentence
        # for movie dataset, we only care about user information: gender, age
        if True:
            user_id = "0"
            for word in input_sentence.split(" "):
                if word.startswith("User_"):
                    user_id = word.replace("User_", "")
                    break
                if word.startswith("user_"):
                    user_id = word.replace("user_", "")
                    break
            assert user_id != "0"
            if "gender" in self.args.feature:
                gender_info = {"F": 0, "M": 1}
                feature = self.user_feature[user_id]["gender"]
                feature = gender_info[feature]
            elif "marital" in self.args.feature:
                marital_info = {
                    "U": 0,
                    "S": 1,
                    "M": 2,
                }
                feature = self.user_feature[user_id]["marital"]
                feature = marital_info[feature]
            elif "occupation" in self.args.feature:
                occupation_info = {
                    "T4MS": 0,
                    "90QI": 1,
                    "56SI": 2,
                }
                feature = self.user_feature[user_id]["occupation"]
                feature = occupation_info[feature]
            elif "age" in self.args.feature:
                feature = self.user_feature[user_id]["age"]
                feature = age_label(feature)

        return input_sentence, target_sentence, feature


def age_label(year):
    if int(year) <= 1959:
        return 0
    elif int(year) <= 1969:
        return 1
    elif int(year) <= 1979:
        return 2
    elif int(year) <= 1989:
        return 3
    else:
        return 4


def occupation_label(number):
    map = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "6": 5,
        "7": 6,
        "10": 7,
        "12": 8,
        "13": 9,
        "14": 10,
        "15": 11,
        "16": 12,
        "17": 13,
        "20": 14,
    }
    if type(number) == int:
        label = map[str(number)]
    else:
        label = map[number]
    return label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--task", type=str, default="movie")
    parser.add_argument("--toy", action="store_true")

    parser.add_argument(
        "--selected_train_data",
        type=str,
        default="adversarial/selected_train_dataset.json",
    )
    parser.add_argument(
        "--selected_test_data",
        type=str,
        default="adversarial/selected_test_dataset.json",
    )
    parser.add_argument(
        "--occupation_train_data",
        type=str,
        default="adversarial/occupation_train_dataset.json",
    )
    parser.add_argument(
        "--occupation_test_data",
        type=str,
        default="adversarial/occupation_test_dataset.json",
    )
    parser.add_argument(
        "--train_data", type=str, default="adversarial/train_dataset.json"
    )
    parser.add_argument(
        "--test_data", type=str, default="adversarial/test_dataset.json"
    )

    ###### insurance dataset ######
    parser.add_argument(
        "--insurance_train_data",
        type=str,
        default="data/insurance/insurance_train_data.json",
    )
    parser.add_argument(
        "--insurance_test_data",
        type=str,
        default="data/insurance/insurance_test_data.json",
    )
    parser.add_argument(
        "--insurance_occupation_train_data",
        type=str,
        default="data/insurance/occupation_train.json",
    )
    parser.add_argument(
        "--insurance_occupation_test_data",
        type=str,
        default="data/insurance/occupation_test.json",
    )
    parser.add_argument(
        "--insurance_marital_train_data",
        type=str,
        default="data/insurance/marital_train.json",
    )
    parser.add_argument(
        "--insurance_marital_test_data",
        type=str,
        default="data/insurance/marital_test.json",
    )
    parser.add_argument(
        "--direct_insurance_train_data",
        type=str,
        default="data/insurance/insurance_direct_train.json",
    )
    parser.add_argument(
        "--direct_insurance_test_data",
        type=str,
        default="data/insurance/insurance_direct_test.json",
    )
    parser.add_argument(
        "--direct_insurance_occupation_train_data",
        type=str,
        default="data/insurance/direct_occupation_train.json",
    )
    parser.add_argument(
        "--direct_insurance_occupation_test_data",
        type=str,
        default="data/insurance/direct_occupation_test.json",
    )
    parser.add_argument(
        "--direct_insurance_marital_train_data",
        type=str,
        default="data/insurance/direct_marital_train.json",
    )
    parser.add_argument(
        "--direct_insurance_marital_test_data",
        type=str,
        default="data/insurance/direct_marital_test.json",
    )

    parser.add_argument("--discriminator_batch_size", type=int, default=1)

    parser.add_argument("--feature", type=str, default="user_gender")

    parser.add_argument("--train_initial_discriminator", action="store_true")
    parser.add_argument("--distributed", action="store_true")

    parser.add_argument("--movie_category_negative_sample", type=int, default=10)
    parser.add_argument("--negative_sample", type=int, default=2)
    parser.add_argument("--sequential_num", type=int, default=25)
    parser.add_argument("--yes_no_sample", type=int, default=5)
    parser.add_argument("--max_history", type=int, default=20)
    parser.add_argument("--direct_item_proportion", type=int, default=2)

    parser.add_argument("--insurance_type", type=str, default="sequential")

    args = parser.parse_args()

    # train_loader, test_loader = load_dataloaders(args)

    # for batch in train_loader:
    #    print(batch)
    #    break

    train, test, user_feature = load_insurance_data(args)

    train_dataset = InsuranceIDInput(args, train, user_feature)

    print(train_dataset[0])
    print(train_dataset[1])
    print(train_dataset[2])
    print(train_dataset[3])
    print(train_dataset[4])

    train_loader, test_loader = load_insurance_dataloaders(args)
    """
    n = 10
    for i, batch in enumerate(train_loader):
        if i < n:
            print(batch)
            print("***")
        else:
            break
    """
