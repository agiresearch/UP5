import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import csv
import time
import numpy as np
from tqdm import tqdm

from transformers import T5Tokenizer
from dataset import calculate_whole_word_ids
from torch.utils.data.distributed import DistributedSampler


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


def load_insurance_dataloaders(args, batch_size=None):
    train, test, user_feature = load_insurance_data(args)

    train_dataset = InsuranceIDInput(args, train, user_feature)
    test_dataset = InsuranceIDInput(args, test, user_feature)

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    collator = Collator(tokenizer)

    if args.distributed:
        sampler = DistributedSampler(train_dataset)
    else:
        sampler = None

    if batch_size == None:
        batch_size = args.discriminator_batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
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
        batch_size=batch_size,
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

    with open(args.insurance_train_data, "r") as f:
        train = json.load(f)
    with open(args.insurance_test_data, "r") as f:
        test = json.load(f)

    random.shuffle(train)

    if args.toy:
        train = train[:100]
        test = train[:10]

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
        marital = d[3]
        occupation = d[7]
        age = d[4]
        user_feature[user_id] = {
            "age": age,
            "marital": marital,
            "occupation": occupation,
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
        label = [input_text[2] for input_text in batch]

        # number of total features, 2 for movie 3 for insurance
        num_of_features = len(label[0])
        ordered_label = [
            [label[i][j] for i in range(len(label))] for j in range(len(label[0]))
        ]
        use_feature = []
        possible_indices = set(list(range(0, num_of_features)))
        for i in range(num_of_features):
            if -1 in ordered_label[i]:
                possible_indices = possible_indices.difference(set([i]))
        # if random.random() < 0.8:
        use_feature = [
            random.choice([0, 1]) if i in possible_indices else 0
            for i in range(num_of_features)
        ]
        if sum(use_feature) == 0:  # == [0] * num_of_features:
            use_feature[random.choice(list(possible_indices))] = 1
        # else:
        #    use_feature = [0] * num_of_features
        #    for i in list(possible_indices):
        #        use_feature[i] = 1

        assert use_feature != [0] * num_of_features

        if num_of_features == 2:
            gender_label = [l[0] for l in label]
            age_label = [l[1] for l in label]
            return (
                torch.tensor(input_ids),
                torch.tensor(input_attention),
                torch.tensor(whole_word_ids),
                torch.tensor(output_ids),
                torch.tensor(output_attention),
                torch.tensor(user_ids),
                torch.tensor(gender_label),
                torch.tensor(age_label),
                torch.tensor(use_feature),
            )
        if num_of_features == 3:
            age_label = [l[0] for l in label]
            marital_label = [l[1] for l in label]
            occupation_label = [l[2] for l in label]
            return (
                torch.tensor(input_ids),
                torch.tensor(input_attention),
                torch.tensor(whole_word_ids),
                torch.tensor(output_ids),
                torch.tensor(output_attention),
                torch.tensor(user_ids),
                torch.tensor(age_label),
                torch.tensor(marital_label),
                torch.tensor(occupation_label),
                torch.tensor(use_feature),
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
        user_id = "0"
        for word in input_sentence.split(" "):
            if word.startswith("user_"):
                user_id = word.replace("user_", "")
                break
        assert user_id != "0"
        gender_info = {"F": 0, "M": 1}
        gender_feature = self.user_feature[user_id]["gender"]
        gender_feature = gender_info[gender_feature]
        age_info = {
            "1": 0,
            "18": 1,
            "25": 2,
            "35": 3,
            "45": 4,
            "50": 5,
            "56": 6,
        }
        age_feature = self.user_feature[user_id]["age"]
        age_feature = age_info[age_feature]

        return input_sentence, target_sentence, [gender_feature, age_feature]


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
        user_id = "0"
        for word in input_sentence.split(" "):
            if word.startswith("User_"):
                user_id = word.replace("User_", "")
                break
            if word.startswith("user_"):
                user_id = word.replace("user_", "")
                break
        assert user_id != "0"

        # marital
        marital_info = {
            "U": 0,
            "S": 1,
            "M": 2,
        }
        marital_feature = self.user_feature[user_id]["marital"]
        if marital_feature in marital_info.keys():
            marital_feature = marital_info[marital_feature]
        else:
            marital_feature = -1
        # occupation
        occupation_info = {
            "T4MS": 0,
            "90QI": 1,
            "56SI": 2,
        }
        occupation_feature = self.user_feature[user_id]["occupation"]
        if occupation_feature in occupation_info.keys():
            occupation_feature = occupation_info[occupation_feature]
        else:
            occupation_feature = -1
        # age
        age_feature = self.user_feature[user_id]["age"]
        age_feature = age_label(age_feature)

        return (
            input_sentence,
            target_sentence,
            [age_feature, marital_feature, occupation_feature],
        )


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
        "--train_data", type=str, default="adversarial/train_dataset.json"
    )
    parser.add_argument(
        "--test_data", type=str, default="adversarial/test_dataset.json"
    )
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

    parser.add_argument("--discriminator_batch_size", type=int, default=2)

    parser.add_argument("--feature", type=str, default="user_gender")

    parser.add_argument("--train_initial_discriminator", action="store_true")
    parser.add_argument("--distributed", action="store_true")

    args = parser.parse_args()

    # train_loader, test_loader = load_dataloaders(args)

    # for batch in train_loader:
    #    print(batch)
    #    break

    train, test, user_feature = load_insurance_data(args)

    train_dataset = InsuranceIDInput(args, train, user_feature)

    # print(train_dataset[0])
    # print(train_dataset[1])
    # print(train_dataset[2])
    # print(train_dataset[3])
    # print(train_dataset[4])

    train_loader, test_loader = load_insurance_dataloaders(args)

    indices = []
    for i, batch in enumerate(tqdm(train_loader)):
        indices.append(batch[-1].tolist())
    print(len([d for d in indices if d == [1, 1, 1]]))
    print(len([d for d in indices if d == [1, 1, 0]]))
    print(len([d for d in indices if d == [1, 0, 1]]))
    print(len([d for d in indices if d == [0, 1, 1]]))
    print(len([d for d in indices if d == [1, 0, 0]]))
    print(len([d for d in indices if d == [0, 0, 1]]))
    print(len([d for d in indices if d == [0, 1, 0]]))
