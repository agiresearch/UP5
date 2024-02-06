from movie_template import task_subgroup_2 as task_sequential_movie
from movie_template import task_subgroup_5 as task_direct_movie
from insurance_template import task_subgroup_2 as task_subgroup_insurance
from insurance_template import task_subgroup_5 as task_direct_insurance
from aliec_template import task_subgroup_2 as task_sequential_aliec
from aliec_template import task_subgroup_5 as task_direct_aliec
from item_representation import Conversion

import argparse
import random
import json
import math
from tqdm import tqdm
import csv
from transformers import T5Tokenizer

from collections import namedtuple
import csv
import math
import json
import random

import argparse


def divide_chunks(k, l, n):
    chunked_list = []
    for i in range(0, len(l), n):
        if len(l[i : i + n]) >= n - 3:
            chunked_list.append([k, l[i : i + n]])
    return chunked_list


def load_insurance_sequential_dataset(args):
    # direct recommendation
    # given chosen insurance, predict the next
    train_all_insurances = []
    user = {"index": "", "gender": "", "marital": "", "occupation": ""}
    recommendation = {"user": {}, "item": [], "time": ""}
    with open(args.data_dir + "insurance/Train.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, l in enumerate(reader):
            bought_insurance = []
            if i == 0:
                insurances = {i: v for i, v in enumerate(l) if i >= 8}
            else:
                if l:
                    user["index"] = l[0]
                    user["gender"] = l[2]
                    user["marital"] = l[3]
                    user["occupation"] = l[7]  # in total 6 occupation type
                    for idx, v in enumerate(l):
                        if v == "1" and idx >= 8:
                            bought_insurance.append(insurances[idx])
                    random.shuffle(bought_insurance)
                    recommendation["user"] = user
                    random.shuffle(bought_insurance)
                    recommendation["item"] = bought_insurance
                    recommendation["time"] = l[1]
                    train_all_insurances.append(recommendation)
                    user = {"index": "", "gender": "", "marital": "", "occupation": ""}
                    recommendation = {"user": {}, "item": [], "time": ""}

    random.shuffle(train_all_insurances)

    train_prop = math.ceil(len(train_all_insurances) * 0.8)
    dev_prop = math.ceil(len(train_all_insurances) * 0.9)
    train_split = train_all_insurances[:train_prop]
    dev_split = train_all_insurances[train_prop:dev_prop]
    test_split = train_all_insurances[dev_prop:]

    if args.toy:
        train_split = train_split[:100]
        dev_split = dev_split[:100]
        test_split = test_split[:100]

    return train_split, dev_split, test_split


def load_insurance_direct_dataset(args):
    # direct recommendation
    # given chosen insurance, predict the next
    train_all_insurances = []
    user_item = {}
    user = {"index": "", "gender": "", "marital": "", "occupation": ""}
    recommendation = {"user": {}, "item": [], "time": ""}
    with open(args.data_dir + "insurance/Train.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, l in enumerate(reader):
            if i == 0:
                insurances = {i: v for i, v in enumerate(l) if i >= 8}
            else:
                if l:
                    bought_insurances = []
                    user["index"] = l[0]
                    user["gender"] = l[2]
                    user["marital"] = l[3]
                    user["occupation"] = l[7]  # in total 6 occupation type
                    for idx, v in enumerate(l):
                        if v == "1" and idx >= 8:
                            recommendation["user"] = user
                            bought_insurances.append(insurances[idx])
                            recommendation["item"] = insurances[idx]
                            recommendation["time"] = l[1]
                            train_all_insurances.append(recommendation)
                            recommendation = {"user": {}, "item": [], "time": ""}
                    user_item[user["index"]] = bought_insurances
                    user = {"index": "", "gender": "", "marital": "", "occupation": ""}
                    recommendation = {"user": {}, "item": [], "time": ""}

    random.shuffle(train_all_insurances)

    train_prop = math.ceil(len(train_all_insurances) * 0.8)
    dev_prop = math.ceil(len(train_all_insurances) * 0.9)
    train_split = train_all_insurances[:train_prop]
    dev_split = train_all_insurances[train_prop:dev_prop]
    test_split = train_all_insurances[dev_prop:]

    if args.toy:
        train_split = train_split[:100]
        dev_split = dev_split[:100]
        test_split = test_split[:100]

    return train_split, dev_split, test_split, user_item


def construct_insurance_dataset(args):
    train, val, test = load_insurance_sequential_dataset(args)

    with open(args.data_dir + "insurance/Train.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, l in enumerate(reader):
            if i == 0:
                insurances = {v for i, v in enumerate(l) if i >= 8}
                break

    # train
    training_data = []
    for input_text, output_text in construct_insurance_sequential_dataset(
        "train", train, insurances
    ):
        training_data.append([input_text, output_text])
    with open(args.data_dir + "insurance/insurance_train.json", "w") as f:
        json.dump(training_data, f)

    # val
    val_data = []
    for input_text, output_text in construct_insurance_sequential_dataset(
        "val", val, insurances
    ):
        val_data.append([template_id, input_text, output_text])
    with open(args.data_dir + "insurance/insurance_val.json", "w") as f:
        json.dump(val_data, f)

    # test
    test_data = []
    for input_text, output_text in construct_insurance_sequential_dataset(
        "test", test, insurances
    ):
        test_data.append([template_id, input_text, output_text])
    with open(args.data_dir + "insurance/insurance_test.json", "w") as f:
        json.dump(test_data, f)


def construct_insurance_sequential_dataset(mode, data, insurances):
    if mode == "train":
        for interaction in tqdm(data):
            user_idx = interaction["user"]["index"]
            bought_insurance = interaction["item"]
            negative_samples = list(set(insurances).difference(set(bought_insurance)))
            if len(bought_insurance) == 1:
                continue
            number_of_sequences = len(bought_insurance) - 1
            for _ in range(number_of_sequences):
                random.shuffle(bought_insurance)
                already_bought = bought_insurance[:-1]
                target_insurance = bought_insurance[-1]
                for template in task_subgroup_insurance:
                    if int(template["id"].split("-")[1]) <= 6:
                        input_text = template["source"].format(
                            user_idx, " , ".join(already_bought)
                        )
                        output_text = template["target"].format(target_insurance)
                    elif int(template["id"].split("-")[1]) <= 10:
                        negative_samples += [target_insurance]
                        random.shuffle(negative_samples)
                        input_text = template["source"].format(
                            user_idx,
                            " , ".join(already_bought),
                            " , ".join(negative_samples),
                        )
                        output_text = template["target"].format(target_insurance)
                    else:
                        for ins in insurances:
                            if ins in bought_insurance and ins == target_insurance:
                                input_text = template["source"].format(
                                    user_idx, " , ".join(already_bought), ins,
                                )
                                output_text = template["target"].format("yes")
                            elif ins not in bought_insurance:
                                input_text = template["source"].format(
                                    user_idx, " , ".join(already_bought), ins
                                )
                                output_text = template["target"].format("no")

                    yield input_text, output_text
    else:
        for interaction in tqdm(data):
            user_idx = interaction["user"]["index"]
            bought_insurance = interaction["item"]
            negative_samples = list(set(insurances).difference(set(bought_insurance)))
            if len(bought_insurance) == 1:
                continue
            random.shuffle(bought_insurance)
            already_bought = bought_insurance[:-1]
            target_insurance = bought_insurance[-1]
            for template in task_subgroup_insurance:
                if int(template["id"].split("-")[1]) <= 6:
                    input_text = template["source"].format(
                        user_idx, " , ".join(already_bought)
                    )
                    output_text = template["target"].format(target_insurance)
                elif int(template["id"].split("-")[1]) <= 10:
                    negative_samples += [target_insurance]
                    random.shuffle(negative_samples)
                    input_text = template["source"].format(
                        user_idx,
                        " , ".join(already_bought),
                        " , ".join(negative_samples),
                    )
                    output_text = template["target"].format(target_insurance)
                else:
                    for ins in insurances:
                        if ins == target_insurance:
                            input_text = template["source"].format(
                                user_idx, " , ".join(already_bought), ins,
                            )
                            output_text = template["target"].format("yes")
                        elif ins not in bought_insurance:
                            input_text = template["source"].format(
                                user_idx, " , ".join(already_bought), ins
                            )
                            output_text = template["target"].format("no")

                yield input_text, output_text


def construct_insurance_2_dataset(args):
    train, val, test, user_item_list = load_insurance_direct_dataset(args)

    with open(args.data_dir + "insurance/Train.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, l in enumerate(reader):
            if i == 0:
                insurances = {v for i, v in enumerate(l) if i >= 8}
                break

    # train
    training_data = []
    for input_text, output_text in construct_insurance_direct_dataset(
        "train", train, insurances, user_item_list
    ):
        training_data.append([input_text, output_text])
    with open(args.data_dir + "insurance/insurance_direct_train.json", "w") as f:
        json.dump(training_data, f)

    # val
    val_data = []
    for input_text, output_text in construct_insurance_direct_dataset(
        "val", val, insurances, user_item_list
    ):
        val_data.append([input_text, output_text])
    with open(args.data_dir + "insurance/insurance_direct_val.json", "w") as f:
        json.dump(val_data, f)

    # test
    test_data = []
    for input_text, output_text in construct_insurance_direct_dataset(
        "test", test, insurances, user_item_list
    ):
        test_data.append([input_text, output_text])
    with open(args.data_dir + "insurance/insurance_direct_test.json", "w") as f:
        json.dump(test_data, f)


def construct_insurance_direct_dataset(mode, data, insurances, user_item_list):
    if mode == "train":
        for interaction in tqdm(data):
            user_idx = interaction["user"]["index"]
            bought_insurance = user_item_list[user_idx]
            target_insurance = interaction["item"]

            negative_samples = list(set(insurances).difference(set(bought_insurance)))
            candidates = [target_insurance] + negative_samples
            random.shuffle(candidates)

            for template in task_direct_insurance:
                if int(template["id"].split("-")[1]) <= 7:
                    if int(template["id"].split("-")[1]) != 2:
                        for i in range(4):
                            if i == 0:
                                input_text = template["source"].format(
                                    user_idx, target_insurance
                                )
                                output_text = template["target"].format("yes")
                                yield input_text, output_text
                            else:
                                input_text = template["source"].format(
                                    user_idx, random.choice(negative_samples)
                                )
                                output_text = template["target"].format("no")
                                yield input_text, output_text
                    else:
                        for i in range(4):
                            if i == 0:
                                input_text = template["source"].format(
                                    target_insurance, user_idx
                                )
                                output_text = template["target"].format("yes")
                                yield input_text, output_text
                            else:
                                input_text = template["source"].format(
                                    user_idx, random.choice(negative_samples)
                                )
                                output_text = template["target"].format("no")
                                yield input_text, output_text
                else:
                    for i in range(4):
                        random.shuffle(candidates)
                        input_text = template["source"].format(
                            user_idx, " , ".join(candidates),
                        )
                        output_text = template["target"].format(target_insurance)
                        yield input_text, output_text
    else:
        for interaction in tqdm(data):
            user_idx = interaction["user"]["index"]
            bought_insurance = user_item_list[user_idx]
            target_insurance = interaction["item"]

            negative_samples = list(set(insurances).difference(set(bought_insurance)))
            candidates = [target_insurance] + negative_samples
            random.shuffle(candidates)

            for template in task_direct_insurance:
                if int(template["id"].split("-")[1]) <= 7:
                    if int(template["id"].split("-")[1]) != 2:
                        for i in range(4):
                            if i == 0:
                                input_text = template["source"].format(
                                    user_idx, target_insurance
                                )
                                output_text = template["target"].format("yes")
                                yield input_text, output_text
                            else:
                                input_text = template["source"].format(
                                    user_idx, random.choice(negative_samples)
                                )
                                output_text = template["target"].format("no")
                                yield input_text, output_text
                    else:
                        for i in range(4):
                            if i == 0:
                                input_text = template["source"].format(
                                    target_insurance, user_idx
                                )
                                output_text = template["target"].format("yes")
                                yield input_text, output_text
                            else:
                                input_text = template["source"].format(
                                    user_idx, random.choice(negative_samples)
                                )
                                output_text = template["target"].format("no")
                                yield input_text, output_text
                else:
                    random.shuffle(candidates)
                    input_text = template["source"].format(
                        user_idx, " , ".join(candidates),
                    )
                    output_text = template["target"].format(target_insurance)
                    yield input_text, output_text



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--sample_division", type=int, default=10)
    parser.add_argument("--max_history", type=int, default=20)
    parser.add_argument("--negative_sample", type=int, default=99)
    parser.add_argument("--negative_sample_times", type=int, default=5)

    parser.add_argument("--use_item_representation", action="store_true")
    parser.add_argument(
        "--item_representation",
        type=str,
        default="textual_numeral",
        help="textual_numeral, hash_table, random_vocab_mapping, vocab_mapping",
    )

    args = parser.parse_args()

    construct_insurance_dataset(args)
    print("finished insurance sequential data construction")
    construct_insurance_2_dataset(args)
    print("finished insurance direct data construction")
