import torch
from transformers import T5Tokenizer, T5Config, T5ForConditionalGenerationwithPrefix
import json
from tqdm import tqdm
import argparse
import os
import random
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
from operator import itemgetter
import heapq
import time
import csv
import sys
from prefix_tuning import (
    PrefixTuningT5,
    AttentionTuningT5,
    ConcatPrompt,
    AveragePrompt,
)

sys.path.append("..")
from dataset import load_dataloaders
from modeling_p5 import P5
from dataset import Collator, InputDataset
from generation_trie import Trie, prefix_allowed_tokens_fn


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


class Logger(object):
    def __init__(self, log_path, on=True):
        self.log_path = log_path
        self.on = on

        if self.on:
            while os.path.isfile(self.log_path):
                self.log_path += "+"

    def log(self, string, newline=True):
        if self.on:
            with open(self.log_path, "a") as logf:
                logf.write(string)
                if newline:
                    logf.write("\n")

            sys.stdout.write(string)
            if newline:
                sys.stdout.write("\n")
            sys.stdout.flush()


def exact_match(predictions, targets):
    correct = 0
    for p, t in zip(predictions, targets):
        if p == t:
            correct += 1

    return correct


def load_model(model, pretrained_dir):
    ckpt = torch.load(pretrained_dir, map_location="cuda:0")
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        k = k.replace("module_", "")
        new_ckpt[k] = v

    # for n, p in model.named_parameters():
    #    print(n)
    # print('***')
    # for k,v in new_ckpt.items():
    #    print(k)
    model.load_state_dict(new_ckpt, strict=False)

    return model


def gender_dict(args):
    data_dir = args.data_dir + args.task + "/"
    data = []
    with open(data_dir + "Train.csv", newline="\n") as f:
        reader = csv.reader(f, delimiter=",")
        for r in reader:
            data.append(r)
    gender_dictionary = {d[0]: d[2] for d in data[1:]}
    return gender_dictionary


def marital_dict(args):
    data_dir = args.data_dir + args.task + "/"
    data = []
    with open(data_dir + "Train.csv", newline="\n") as f:
        reader = csv.reader(f, delimiter=",")
        for r in reader:
            data.append(r)
    marital_dictionary = {d[0]: d[3] for d in data[1:]}
    return marital_dictionary


def occupation_dict(args):
    data_dir = args.data_dir + args.task + "/"
    data = []
    with open(data_dir + "Train.csv", newline="\n") as f:
        reader = csv.reader(f, delimiter=",")
        for r in reader:
            data.append(r)
    occupation_dictionary = {d[0]: d[7] for d in data[1:]}
    return occupation_dictionary


def age_dict(args):
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

    data_dir = args.data_dir + args.task + "/"
    data = []
    with open(data_dir + "Train.csv", newline="\n") as f:
        reader = csv.reader(f, delimiter=",")
        for r in reader:
            data.append(r)
    age_dictionary = {d[0]: age_label(d[4]) for d in data[1:]}
    return age_dictionary


def select_gender(input_text, gender, gender_dict):
    for word in input_text.split(" "):
        if word.startswith("User_"):
            word = word.replace("User_", "")
            return gender_dict[word] == gender


# M, S, U, W, D
def select_marital(input_text, marital, marital_dict):
    for word in input_text.split(" "):
        if word.startswith("User_"):
            word = word.replace("User_", "")
            return marital_dict[word] == marital


def select_occupation(input_text, occupation, occupation_dict):
    for word in input_text.split(" "):
        if word.startswith("User_"):
            word = word.replace("User_", "")
            return occupation_dict[word] == occupation


def select_age(input_text, age, age_dict):
    for word in input_text.split(" "):
        if word.startswith("User_"):
            word = word.replace("User_", "")
            return age_dict[word] == age


def load_dataloaders(args, tokenizer, template_id, logger):
    data_dir = args.data_dir + args.task + "/"

    if "marital" in args.feature:
        test_dir = data_dir + "direct_marital_test.json"
    elif "age" in args.feature:
        test_dir = data_dir + "insurance_direct_test.json"
    else:
        assert "occupation" in args.feature
        test_dir = data_dir + "direct_occupation_test.json"

    with open(test_dir, "r") as f:
        test = json.load(f)

    collator = Collator(tokenizer)

    if template_id == "5-8":
        test = [
            [data[0], data[1]]
            for data in test
            if "Which insurance of the following to recommend for user_" in data[0]
        ]
    elif template_id == "5-9":
        test = [
            [data[0], data[1]]
            for data in test
            if "Choose the best insurance from the candidates to recommend for user_"
            in data[0]
        ]
    elif template_id == "5-10":
        test = [
            [data[0], data[1]]
            for data in test
            if "Pick the most useful insurance from the following list and recommend to user_"
            in data[0]
        ]

    type_id = int(template_id.split("-")[1])

    if type_id <= 7:
        logger.log("yes or no question")
    else:
        logger.log("direct recommendation")

    logger.log("example datapoint:")
    logger.log("input is {}".format(test[0][0]))
    logger.log("output target is {}".format(test[0][1]))

    if not args.compute_parity:
        testdataset = InputDataset(args, test)

        test_loader = DataLoader(
            testdataset, batch_size=args.batch_size, collate_fn=collator, shuffle=False,
        )

        return test_loader

    else:
        if args.feature == "user_gender":
            gender_dictionary = gender_dict(args)
            female_test_data = [
                d for d in test if select_gender(d[0], "F", gender_dictionary)
            ]
            male_test_data = [
                d for d in test if select_gender(d[0], "M", gender_dictionary)
            ]

            maletestdataset = InputDataset(args, male_test_data)
            femaletestdataset = InputDataset(args, female_test_data)

            male_test_loader = DataLoader(
                maletestdataset,
                batch_size=args.batch_size,
                collate_fn=collator,
                shuffle=False,
            )
            female_test_loader = DataLoader(
                femaletestdataset,
                batch_size=args.batch_size,
                collate_fn=collator,
                shuffle=False,
            )

            return (
                male_test_loader,
                female_test_loader,
            )
        elif args.feature == "user_age":
            age_dictionary = age_dict(args)
            one_test_data = [d for d in test if select_age(d[0], 0, age_dictionary)]
            two_test_data = [d for d in test if select_age(d[0], 1, age_dictionary)]
            three_test_data = [d for d in test if select_age(d[0], 2, age_dictionary)]
            four_test_data = [d for d in test if select_age(d[0], 3, age_dictionary)]
            five_test_data = [d for d in test if select_age(d[0], 4, age_dictionary)]

            onetestdataset = InputDataset(args, one_test_data)
            twotestdataset = InputDataset(args, two_test_data)
            threetestdataset = InputDataset(args, three_test_data)
            fourtestdataset = InputDataset(args, four_test_data)
            fivetestdataset = InputDataset(args, five_test_data)

            one_test_loader = DataLoader(
                onetestdataset,
                batch_size=args.batch_size,
                collate_fn=collator,
                shuffle=False,
            )
            two_test_loader = DataLoader(
                twotestdataset,
                batch_size=args.batch_size,
                collate_fn=collator,
                shuffle=False,
            )
            three_test_loader = DataLoader(
                threetestdataset,
                batch_size=args.batch_size,
                collate_fn=collator,
                shuffle=False,
            )
            four_test_loader = DataLoader(
                fourtestdataset,
                batch_size=args.batch_size,
                collate_fn=collator,
                shuffle=False,
            )
            five_test_loader = DataLoader(
                fivetestdataset,
                batch_size=args.batch_size,
                collate_fn=collator,
                shuffle=False,
            )

            return (
                one_test_loader,
                two_test_loader,
                three_test_loader,
                four_test_loader,
                five_test_loader,
            )
        # M, S, U, W, D
        elif args.feature == "user_marital":
            marital_dictionary = marital_dict(args)

            M_test_data = [
                d for d in test if select_marital(d[0], "M", marital_dictionary)
            ]
            S_test_data = [
                d for d in test if select_marital(d[0], "S", marital_dictionary)
            ]
            U_test_data = [
                d for d in test if select_marital(d[0], "U", marital_dictionary)
            ]
            W_test_data = [
                d for d in test if select_marital(d[0], "W", marital_dictionary)
            ]
            D_test_data = [
                d for d in test if select_marital(d[0], "D", marital_dictionary)
            ]

            Mtestdataset = InputDataset(args, M_test_data)
            Stestdataset = InputDataset(args, S_test_data)
            Utestdataset = InputDataset(args, U_test_data)
            Wtestdataset = InputDataset(args, W_test_data)
            Dtestdataset = InputDataset(args, D_test_data)

            M_test_loader = DataLoader(
                Mtestdataset,
                batch_size=args.batch_size,
                collate_fn=collator,
                shuffle=False,
            )
            S_test_loader = DataLoader(
                Stestdataset,
                batch_size=args.batch_size,
                collate_fn=collator,
                shuffle=False,
            )
            U_test_loader = DataLoader(
                Utestdataset,
                batch_size=args.batch_size,
                collate_fn=collator,
                shuffle=False,
            )
            W_test_loader = DataLoader(
                Wtestdataset,
                batch_size=args.batch_size,
                collate_fn=collator,
                shuffle=False,
            )
            D_test_loader = DataLoader(
                Dtestdataset,
                batch_size=args.batch_size,
                collate_fn=collator,
                shuffle=False,
            )

            return (
                M_test_loader,
                S_test_loader,
                U_test_loader,
                W_test_loader,
                D_test_loader,
            )
        # M, S, U, W, D
        elif args.feature == "user_occupation":
            occupation_dictionary = occupation_dict(args)
            A_test_data = [
                d
                for d in test
                if select_occupation(d[0], "L44T", occupation_dictionary)
            ]
            B_test_data = [
                d
                for d in test
                if select_occupation(d[0], "56SI", occupation_dictionary)
            ]
            C_test_data = [
                d
                for d in test
                if select_occupation(d[0], "T4MS", occupation_dictionary)
            ]
            D_test_data = [
                d
                for d in test
                if select_occupation(d[0], "JD7X", occupation_dictionary)
            ]
            E_test_data = [
                d
                for d in test
                if select_occupation(d[0], "90QI", occupation_dictionary)
            ]

            Atestdataset = InputDataset(args, A_test_data)
            Btestdataset = InputDataset(args, B_test_data)
            Ctestdataset = InputDataset(args, C_test_data)
            Dtestdataset = InputDataset(args, D_test_data)
            Etestdataset = InputDataset(args, E_test_data)

            A_test_loader = DataLoader(
                Atestdataset,
                batch_size=args.batch_size,
                collate_fn=collator,
                shuffle=False,
            )
            B_test_loader = DataLoader(
                Btestdataset,
                batch_size=args.batch_size,
                collate_fn=collator,
                shuffle=False,
            )
            C_test_loader = DataLoader(
                Ctestdataset,
                batch_size=args.batch_size,
                collate_fn=collator,
                shuffle=False,
            )
            D_test_loader = DataLoader(
                Dtestdataset,
                batch_size=args.batch_size,
                collate_fn=collator,
                shuffle=False,
            )
            E_test_loader = DataLoader(
                Etestdataset,
                batch_size=args.batch_size,
                collate_fn=collator,
                shuffle=False,
            )

            return (
                A_test_loader,
                B_test_loader,
                C_test_loader,
                D_test_loader,
                E_test_loader,
            )


def direct_sample_100(args, prefix_model, model, batch, tokenizer):
    candidates = []
    with open(args.data_dir + args.task + "/" + "Train.csv", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        for r in reader:
            candidates.append(r)
            break
    candidates = candidates[0][8:]

    input_ids = batch[0].cuda()
    attn = batch[1].cuda()
    whole_input_ids = batch[2].cuda()
    output_ids = batch[3].tolist()

    hits1 = 0
    hits3 = 0
    hits5 = 0
    total = 0
    hits1_result = []
    hits3_result = []
    hits5_result = []
    for b in range(input_ids.size(0)):
        # compute candidate trie
        input_sentence = tokenizer.convert_ids_to_tokens(input_ids[b])
        input_sentence = [word.replace("▁", " ") for word in input_sentence]
        input_sentence = "".join(input_sentence)
        input_sentence = input_sentence.replace("</s>", "").replace("<pad>", "")

        sentence_candidates = [c for c in candidates if c in input_sentence]
        candidate_trie = Trie(
            [[0] + tokenizer.encode("{}".format(e)) for e in sentence_candidates]
        )
        prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

        model_inputs = {
            "input_ids": input_ids[b : b + 1],
            "whole_word_ids": whole_input_ids[b : b + 1],
            "attention_mask": attn[b : b + 1],
        }
        top_five_ids = prefix_model.generate(
            model,
            **model_inputs,
            max_length=10,
            prefix_allowed_tokens_fn=prefix_allowed_tokens,
            num_beams=20,
            num_return_sequences=5,
        )
        top_five = tokenizer.batch_decode(top_five_ids, skip_special_tokens=True)
        top_three = top_five[:3]
        top_one = top_five[:1]

        # length
        gold_id = tokenizer.decode(output_ids[b], skip_special_tokens=True)
        if gold_id in top_one:
            hits1 += 1
        if gold_id in top_three:
            hits3 += 1
        if gold_id in top_five:
            hits5 += 1
        total += 1
        hits1_result.append([top_one, gold_id])
        hits3_result.append([top_three, gold_id])
        hits5_result.append([top_five, gold_id])

    return hits1, hits3, hits5, total, hits1_result, hits3_result, hits5_result


def direct_yes_no(prefix_model, model, batch, tokenizer):
    total = 0
    correct = 0
    input_ids = batch[0].cuda()
    attn = batch[1].cuda()
    whole_input_ids = batch[2].cuda()
    output_ids = batch[3].tolist()
    output_attention = batch[4].cuda()

    candidates = ["yes", "no"]
    candidate_trie = Trie([[0] + tokenizer.encode("{}".format(e)) for e in candidates])
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

    model_inputs = {
        "input_ids": input_ids,
        "whole_word_ids": whole_input_ids,
        "attention_mask": attn,
    }
    generated_ids = prefix_model.generate(
        model,
        **model_inputs,
        max_length=10,
        prefix_allowed_tokens_fn=prefix_allowed_tokens,
    )
    generated_answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    results = []
    # B, length
    gold_sents = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    for gold_sent, predicted_sent in zip(gold_sents, generated_answer):
        if gold_sent == predicted_sent:
            correct += 1
        total += 1
        results.append([predicted_sent, gold_sent])

    return correct, total, results


def evaluate_one_dataloder(args, logger, loader, prefix_model, model, tokenizer):
    correct = 0
    total = 0
    hits1 = 0
    hits3 = 0
    hits5 = 0
    collect_results = []
    collect_results_1 = []
    collect_results_3 = []
    collect_results_5 = []
    with torch.no_grad():
        for batch in tqdm(loader):
            if int(args.template_id.split("-")[0]) == 5:
                if int(args.template_id.split("-")[1]) >= 8:
                    (
                        batch_hits1,
                        batch_hits3,
                        batch_hits5,
                        batch_total,
                        hits1_result,
                        hits3_result,
                        hits5_result,
                    ) = direct_sample_100(args, prefix_model, model, batch, tokenizer)
                    hits1 += batch_hits1
                    hits3 += batch_hits3
                    hits5 += batch_hits5
                    total += batch_total
                    collect_results_1 += hits1_result
                    collect_results_3 += hits3_result
                    collect_results_5 += hits5_result
                else:
                    batch_correct, batch_total, result = direct_yes_no(
                        prefix_model, model, batch, tokenizer
                    )
                    correct += batch_correct
                    total += batch_total
                    collect_results += result

    if int(args.template_id.split("-")[0]) == 5:
        if int(args.template_id.split("-")[1]) >= 8:
            hits1 = hits1 / total
            hits3 = hits3 / total
            hits5 = hits5 / total
            logger.log(
                "hits@1 for {} is {}, hits@3 is {}, hits@5 is {}".format(
                    args.template_id, hits1, hits3, hits5
                )
            )
        else:
            accuracy = correct / total
            logger.log("accuracy for {} is {}".format(args.template_id, accuracy))


def evaluate_and_save(args, logger):
    logger.log("load model")
    config = T5Config.from_pretrained("t5-small")
    config.initialization = "normal"
    model = T5ForConditionalGenerationwithPrefix.from_pretrained("t5-small").cuda()

    if not args.combine_prompts:
        logger.log("use only one prompt")
        if args.use_attention:
            prefix_model = AttentionTuningT5(
                config=config,
                preseqlen=args.prefix_length,
                attnseqlen=args.attn_prefix_length,
            ).cuda()
        else:
            prefix_model = PrefixTuningT5(
                config=config, preseqlen=args.prefix_length
            ).cuda()
        prefix_model.load_state_dict(torch.load(args.prefix_pretrained_dir))
        model = load_model(model, args.P5_pretrained_dir)

        logger.log("load data")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        if not args.compute_parity:
            test_loader = load_dataloaders(args, tokenizer, args.template_id, logger)
            logger.log("start evaluation")
            evaluate_one_dataloder(
                args, logger, test_loader, prefix_model, model, tokenizer
            )
        else:
            if args.feature == "user_gender":
                (male_test_loader, female_test_loader,) = load_dataloaders(
                    args, tokenizer, args.template_id, logger
                )
                logger.log("start evaluation on male data")
                evaluate_one_dataloder(
                    args, logger, male_test_loader, prefix_model, model, tokenizer
                )
                logger.log("start evaluation on female data")
                evaluate_one_dataloder(
                    args, logger, female_test_loader, prefix_model, model, tokenizer
                )
            if args.feature == "user_age":
                (
                    one_test_loader,
                    two_test_loader,
                    three_test_loader,
                    four_test_loader,
                    five_test_loader,
                ) = load_dataloaders(args, tokenizer, args.template_id, logger)
                logger.log("start evaluation on one data")
                evaluate_one_dataloder(
                    args, logger, one_test_loader, prefix_model, model, tokenizer
                )
                logger.log("start evaluation on two data")
                evaluate_one_dataloder(
                    args, logger, two_test_loader, prefix_model, model, tokenizer
                )
                logger.log("start evaluation on three data")
                evaluate_one_dataloder(
                    args, logger, three_test_loader, prefix_model, model, tokenizer
                )
                logger.log("start evaluation on four data")
                evaluate_one_dataloder(
                    args, logger, four_test_loader, prefix_model, model, tokenizer
                )
                logger.log("start evaluation on five data")
                evaluate_one_dataloder(
                    args, logger, five_test_loader, prefix_model, model, tokenizer
                )
            elif args.feature == "user_marital":
                (
                    M_test_loader,
                    S_test_loader,
                    U_test_loader,
                    W_test_loader,
                    D_test_loader,
                ) = load_dataloaders(args, tokenizer, args.template_id, logger)

                logger.log("start evaluation on married data")
                evaluate_one_dataloder(
                    args, logger, M_test_loader, prefix_model, model, tokenizer
                )
                logger.log("start evaluation on separated data")
                evaluate_one_dataloder(
                    args, logger, S_test_loader, prefix_model, model, tokenizer
                )
                logger.log("start evaluation on unmarried data")
                evaluate_one_dataloder(
                    args, logger, U_test_loader, prefix_model, model, tokenizer
                )
                logger.log("start evaluation on windowed data")
                evaluate_one_dataloder(
                    args, logger, W_test_loader, prefix_model, model, tokenizer
                )
                logger.log("start evaluation on divorced data")
                evaluate_one_dataloder(
                    args, logger, D_test_loader, prefix_model, model, tokenizer
                )
            elif args.feature == "user_occupation":
                (
                    A_test_loader,
                    B_test_loader,
                    C_test_loader,
                    D_test_loader,
                    E_test_loader,
                ) = load_dataloaders(args, tokenizer, args.template_id, logger)
                logger.log("start evaluation on L44T data")
                evaluate_one_dataloder(
                    args, logger, A_test_loader, prefix_model, model, tokenizer
                )
                logger.log("start evaluation on 56SI data")
                evaluate_one_dataloder(
                    args, logger, B_test_loader, prefix_model, model, tokenizer
                )
                logger.log("start evaluation on T4MS data")
                evaluate_one_dataloder(
                    args, logger, C_test_loader, prefix_model, model, tokenizer
                )
                logger.log("start evaluation on JD7X data")
                evaluate_one_dataloder(
                    args, logger, D_test_loader, prefix_model, model, tokenizer
                )
                logger.log("start evaluation on 90QI data")
                evaluate_one_dataloder(
                    args, logger, E_test_loader, prefix_model, model, tokenizer
                )

    else:
        logger.log("use multiple prompts combined")
        prefix_model_gender = PrefixTuningT5(config).cuda()
        prefix_model_age = PrefixTuningT5(config).cuda()
        prefix_model_gender.load_state_dict(
            torch.load(args.gender_prefix_pretrained_dir)
        )
        prefix_model_age.load_state_dict(torch.load(args.age_prefix_pretrained_dir))
        model = load_model(model, args.P5_pretrained_dir)
        if args.combine_method == "concatenation":
            multi_prompt = ConcatPrompt([prefix_model_age, prefix_model_gender])
        elif args.combine_method == "average":
            multi_prompt = AveragePrompt([prefix_model_age, prefix_model_gender])

        logger.log("load data")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        if not args.compute_parity:
            val_loader, test_loader = load_dataloaders(
                args, tokenizer, args.template_id, logger
            )
            logger.log("start evaluation")
            evaluate_one_dataloder(
                args, logger, test_loader, multi_prompt, model, tokenizer
            )
        else:
            (
                male_val_loader,
                female_val_loader,
                male_test_loader,
                female_test_loader,
            ) = load_dataloaders(args, tokenizer, args.template_id, logger)

            logger.log("start evaluation")
            logger.log("compute male data score")
            evaluate_one_dataloder(
                args, logger, male_test_loader, multi_prompt, model, tokenizer
            )
            logger.log("compute female data score")
            evaluate_one_dataloder(
                args, logger, female_test_loader, multi_prompt, model, tokenizer
            )


def evaluate_all(args):
    logger.log("load model")
    config = T5Config.from_pretrained("t5-small")
    config.initialization = "normal"
    model = T5ForConditionalGenerationwithPrefix.from_pretrained("t5-small").cuda()
    prefix_model = PrefixTuningT5(config)
    model.load_state_dict(torch.load(args.prefix_pretrained_dir))
    model = load_model(model, args.P5_pretrained_dir)

    test_dir = (
        args.data_dir
        + args.task
        + "/"
        + args.item_representation
        + "/"
        + args.task
        + "_val_data.json"
    )
    with open(test_dir, "r") as f:
        test = json.load(f)

    testdataset = InputDataset(args, test)
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    collator = Collator(tokenizer)
    test_loader = DataLoader(
        testdataset, batch_size=args.batch_size, collate_fn=collator, shuffle=False,
    )
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch[0].cuda()
            attn = batch[1].cuda()
            whole_input_ids = batch[2].cuda()
            output_ids = batch[3].cuda()
            output_attention = batch[4].cuda()

            inputs = {"whole_word_ids": whole_input_ids, "input_ids": input_ids}
            prediction = prefix_model.generate(
                model, **inputs, attention_mask=attn, max_length=8
            )

            gold_sents = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            generated_sents = tokenizer.batch_decode(
                prediction, skip_special_tokens=True
            )

            correct += exact_match(generated_sents, gold_sents)
            total += len(gold_sents)
    EM = correct / total
    logger.log("EM is {}".format(EM))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--task", type=str, default="insurance")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gpu", type=str, default="7")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--distributed", action="store_true")

    parser.add_argument("--prefix_pretrained_dir", type=str)
    parser.add_argument("--P5_pretrained_dir", type=str)

    parser.add_argument("--use_item_representation", action="store_true")
    parser.add_argument(
        "--item_representation",
        type=str,
        default="random_vocab_mapping",
        help="hash_table, random_vocab_mapping, vocab_mapping",
    )

    parser.add_argument("--template_id", type=str, default="5-10")

    parser.add_argument("--compute_parity", action="store_true")
    parser.add_argument("--feature", type=str, default="age")

    parser.add_argument("--combine_prompts", action="store_true")
    parser.add_argument("--combine_method", type=str, default="concatenation")
    parser.add_argument("--gender_prefix_pretrained_dir", type=str)
    parser.add_argument("--age_prefix_pretrained_dir", type=str)
    parser.add_argument("--marital_prefix_pretrained_dir", type=str)
    parser.add_argument("--occupation_prefix_pretrained_dir", type=str)

    parser.add_argument("--prefix_length", type=int, default=1)
    parser.add_argument("--attn_prefix_length", type=int, default=5)
    parser.add_argument("--use_attention", action="store_true")

    parser.add_argument(
        "--logging_dir", type=str, default="adversarial/adversarial.log"
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_seed(args)
    logger = Logger(args.logging_dir, True)
    logger.log(str(args))

    evaluate_and_save(args, logger)

    # evaluate_all(args)
