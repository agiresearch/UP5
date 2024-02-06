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
import sys
from prefix_tuning import (
    AttentionTuningT5,
    PrefixTuningT5,
    ConcatPrompt,
    AveragePrompt,
    CFunctionPrompt,
    AttnFFNPrefixTuningT5,
)

sys.path.append("..")
from modeling_p5 import P5
from dataset import Collator, InputDataset
from generation_trie import Trie, prefix_allowed_tokens_fn
from evaluate_and_save import load_new_dataloader, load_new_parity_dataloader


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


def compute_gender(user_info, alldata):
    male_data = []
    female_data = []
    for data in alldata:
        gender = None
        input_ = data[0]
        for word in input_.split(" "):
            if word.startswith("user_"):
                gender = user_info[word.replace("user_", "")]["gender"]
                if gender == "M":
                    male_data.append(data)
                else:
                    female_data.append(data)
                break
    return male_data, female_data


def compute_age(user_info, alldata):
    one_data = []
    two_data = []
    three_data = []
    four_data = []
    five_data = []
    six_data = []
    seven_data = []
    for data in alldata:
        gender = None
        input_ = data[0]
        for word in input_.split(" "):
            if word.startswith("user_"):
                gender = user_info[word.replace("user_", "")]["age"]
                if gender == "1":
                    one_data.append(data)
                elif gender == "18":
                    two_data.append(data)
                elif gender == "25":
                    three_data.append(data)
                elif gender == "35":
                    four_data.append(data)
                elif gender == "45":
                    five_data.append(data)
                elif gender == "50":
                    six_data.append(data)
                else:
                    seven_data.append(data)
                break
    return (one_data, two_data, three_data, four_data, five_data, six_data, seven_data)


def load_dataloaders(args, logger, tokenizer, template_id):
    if not args.compute_parity:
        val_loader, test_loader = load_new_dataloader(args, tokenizer, template_id)
        return val_loader, test_loader
    else:
        if "gender" in args.feature:
            (
                male_val_loader,
                female_val_loader,
                male_test_loader,
                female_test_loader,
            ) = load_new_parity_dataloader(args, tokenizer, template_id)
            return (
                male_val_loader,
                female_val_loader,
                male_test_loader,
                female_test_loader,
            )
        elif "age" in args.feature:
            (
                one_val_loader,
                two_val_loader,
                three_val_loader,
                four_val_loader,
                five_val_loader,
                six_val_loader,
                seven_val_loader,
                one_test_loader,
                two_test_loader,
                three_test_loader,
                four_test_loader,
                five_test_loader,
                six_test_loader,
                seven_test_loader,
            ) = load_new_parity_dataloader(args, tokenizer, template_id)
            return (
                one_val_loader,
                two_val_loader,
                three_val_loader,
                four_val_loader,
                five_val_loader,
                six_val_loader,
                seven_val_loader,
                one_test_loader,
                two_test_loader,
                three_test_loader,
                four_test_loader,
                five_test_loader,
                six_test_loader,
                seven_test_loader,
            )


def sequential_no_sampling(prefix_model, model, batch, tokenizer):
    total = 0
    correct = 0
    input_ids = batch[0].cuda()
    attn = batch[1].cuda()
    whole_input_ids = batch[2].cuda()
    output_ids = batch[3].tolist()
    output_attention = batch[4].cuda()

    model_inputs = {
        "input_ids": input_ids,
        "whole_word_ids": whole_input_ids,
        "attention_mask": attn,
    }

    batch_size = input_ids.size(0)
    predictions = prefix_model.generate(
        model=model,
        **model_inputs,
        max_length=10,
        num_beams=20,
        num_return_sequences=10,
        output_scores=True,
        return_dict_in_generate=True,
    )
    # B, num_return_sequences, max_length
    predicted_sequences = predictions["sequences"].view(batch_size, 10, -1).tolist()
    predicted_scores = predictions["sequences_scores"].view(batch_size, 10, -1)

    # B, length
    gold_sents = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    results = []
    for gold_sent, predicted_sent in zip(gold_sents, predicted_sequences):
        # num_return_sequences, max_length
        predicted_sentences = tokenizer.batch_decode(
            predicted_sent, skip_special_tokens=True
        )
        if gold_sent in predicted_sentences:
            correct += 1
        total += 1
        results.append([predicted_sentences, gold_sent])

    return correct, total, results


def sequential_random_sampling(prefix_model, model, batch, tokenizer):
    hits1 = 0
    hits3 = 0
    hits5 = 0
    hits10 = 0
    total = 0
    hits1_result = []
    hits3_result = []
    hits5_result = []
    hits10_result = []

    # requires batch_size = 1
    input_ids = batch[0].cuda()
    attn = batch[1].cuda()
    whole_input_ids = batch[2].cuda()
    output_ids = batch[3].tolist()
    output_attention = batch[4].cuda()

    gold_output = tokenizer.decode(output_ids[0][:-1])
    gold_output = gold_output.replace("</s>", "")
    all_candidates = list(range(6040))
    all_candidates.remove(int(gold_output))
    candidates = [gold_output] + [str(x) for x in random.sample(all_candidates, k=100)]
    candidate_trie = Trie([[0] + tokenizer.encode("{}".format(e)) for e in candidates])
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

    model_inputs = {
        "input_ids": input_ids,
        "whole_word_ids": whole_input_ids,
        "attention_mask": attn,
    }

    top_ten_ids = prefix_model.generate(
        model=model,
        **model_inputs,
        max_length=10,
        prefix_allowed_tokens_fn=prefix_allowed_tokens,
        num_beams=20,
        num_return_sequences=10,
    )
    top_ten = tokenizer.batch_decode(top_ten_ids, skip_special_tokens=True)
    top_five = top_ten[:5]
    top_three = top_ten[:3]
    top_one = top_ten[:1]

    # length
    gold_id = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if gold_id in top_one:
        hits1 += 1
    if gold_id in top_three:
        hits3 += 1
    if gold_id in top_five:
        hits5 += 1
    if gold_id in top_ten:
        hits10 += 1
    total += 1
    hits1_result.append([top_one, gold_id])
    hits3_result.append([top_three, gold_id])
    hits5_result.append([top_five, gold_id])
    hits10_result.append([top_ten, gold_id])

    return (
        hits1,
        hits3,
        hits5,
        hits10,
        total,
        hits1_result,
        hits3_result,
        hits5_result,
        hits10_result,
    )


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


def find_candidates(sentence):
    if " : " in sentence:
        if sentence.count(" : ") == 1:
            candidates = sentence[sentence.index(": ") + 1 :].split(" , ")
        else:
            indices = [i for i, c in enumerate(sentence) if c == ": "]
            last_index = indices[-1]
            candidates = sentence[last_index + 1 :].split(" , ")
    elif " ? " in sentence:
        candidates = sentence[sentence.index("?") + 1 :].split(" , ")
    else:
        logger.log("input format wrong")

    candidates = [c.strip() for c in candidates]

    return candidates


def find_topk_candidates(loss, candidates):
    loss = [-1 * l for l in loss]
    one_val = heapq.nlargest(1, enumerate(loss), key=itemgetter(1))
    one_indices = [i for (i, val) in sorted(one_val)]
    top_one = [candidates[i] for i in one_indices]
    three_val = heapq.nlargest(3, enumerate(loss), key=itemgetter(1))
    three_indices = [i for (i, val) in sorted(three_val)]
    top_three = [candidates[i] for i in three_indices]
    five_val = heapq.nlargest(5, enumerate(loss), key=itemgetter(1))
    five_indices = [i for (i, val) in sorted(five_val)]
    top_five = [candidates[i] for i in five_indices]

    return top_one, top_three, top_five


def direct_sample_100(args, prefix_model, model, batch, tokenizer):
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
        candidates = find_candidates(input_sentence)
        assert len(candidates) == 101
        candidate_trie = Trie(
            [[0] + tokenizer.encode("{}".format(e)) for e in candidates]
        )
        prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

        if not args.combine_prompts:
            model_inputs = {
                "input_ids": input_ids[b : b + 1],
                "whole_word_ids": whole_input_ids[b : b + 1],
                "attention_mask": attn[b : b + 1],
            }
        else:
            model_inputs = {
                "input_ids": input_ids[b : b + 1],
                "whole_word_ids": whole_input_ids[b : b + 1],
                "attention_mask": attn[b : b + 1],
                "model_indices": [1, 1],
            }
        top_five_ids = prefix_model.generate(
            model=model,
            **model_inputs,
            max_length=10,
            prefix_allowed_tokens_fn=prefix_allowed_tokens,
            num_beams=20,
            num_return_sequences=5,
        )
        top_five = tokenizer.batch_decode(top_five_ids, skip_special_tokens=True)
        top_three_ids = prefix_model.generate(
            model=model,
            **model_inputs,
            max_length=10,
            prefix_allowed_tokens_fn=prefix_allowed_tokens,
            num_beams=20,
            num_return_sequences=3,
        )
        top_three = tokenizer.batch_decode(top_three_ids, skip_special_tokens=True)
        top_one_ids = prefix_model.generate(
            model=model,
            **model_inputs,
            max_length=10,
            prefix_allowed_tokens_fn=prefix_allowed_tokens,
            num_beams=20,
            num_return_sequences=1,
        )
        top_one = tokenizer.batch_decode(top_one_ids, skip_special_tokens=True)

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


def evaluate_one_dataloder(args, logger, loader, prefix_model, model, tokenizer):
    correct = 0
    total = 0
    hits1 = 0
    hits3 = 0
    hits5 = 0
    hits10 = 0
    collect_results = []
    collect_results_1 = []
    collect_results_3 = []
    collect_results_5 = []
    collect_results_10 = []
    with torch.no_grad():
        for batch in tqdm(loader):
            if int(args.template_id.split("-")[0]) == 2:
                if int(args.template_id.split("-")[1]) <= 6:
                    (
                        batch_hits1,
                        batch_hits3,
                        batch_hits5,
                        batch_hits10,
                        batch_total,
                        hits1_result,
                        hits3_result,
                        hits5_result,
                        hits10_result,
                    ) = sequential_random_sampling(
                        prefix_model, model, batch, tokenizer
                    )
                    hits1 += batch_hits1
                    hits3 += batch_hits3
                    hits5 += batch_hits5
                    hits10 += batch_hits10
                    total += batch_total
                elif int(args.template_id.split("-")[1]) <= 10:
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
            elif int(args.template_id.split("-")[1]) <= 7:
                batch_correct, batch_total, result = direct_yes_no(
                    prefix_model, model, batch, tokenizer
                )
                correct += batch_correct
                total += batch_total
                collect_results += result
            else:
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

    if int(args.template_id.split("-")[0]) == 2:
        if int(args.template_id.split("-")[1]) <= 6:
            hits1 = hits1 / total
            hits3 = hits3 / total
            hits5 = hits5 / total
            hits10 = hits10 / total
            logger.log(
                "hits@1 for {} is {},hits@3 is {}, hits@5 is {}, hits@10 is {}".format(
                    args.template_id, hits1, hits3, hits5, hits10
                )
            )
        elif int(args.template_id.split("-")[1]) <= 10:
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
    elif int(args.template_id.split("-")[1]) <= 7:
        accuracy = correct / total
        logger.log("accuracy for {} is {}".format(args.template_id, accuracy))
        with open("{}_yes_no_result.json".format(args.template_id), "w") as f:
            json.dump(collect_results, f)
    else:
        hits1 = hits1 / total
        hits3 = hits3 / total
        hits5 = hits5 / total
        logger.log(
            "hits@1 for {} is {},hits@3 is {}, hits@5 is {}".format(
                args.template_id, hits1, hits3, hits5
            )
        )


def evaluate_and_save(args, logger):
    logger.log("load model")
    config = T5Config.from_pretrained("t5-small")
    config.initialization = (
        "zero" if args.task == "movie" and "age" in args.feature else "normal"
    )
    model = T5ForConditionalGenerationwithPrefix.from_pretrained("t5-small").cuda()

    if not args.combine_prompts:
        logger.log("use only one prompt")
        if not args.use_attention:
            prefix_model = PrefixTuningT5(
                config=config, preseqlen=args.prefix_length
            ).cuda()
        else:
            prefix_model = AttentionTuningT5(
                config=config,
                preseqlen=args.prefix_length,
                attnseqlen=args.attn_prefix_length,
            ).cuda()
        prefix_model.load_state_dict(torch.load(args.prefix_pretrained_dir))
        model = load_model(model, args.P5_pretrained_dir)

        logger.log("load data")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        if not args.compute_parity:
            val_loader, test_loader = load_dataloaders(
                args, logger, tokenizer, args.template_id
            )
            logger.log("start evaluation")
            evaluate_one_dataloder(
                args, logger, test_loader, prefix_model, model, tokenizer
            )
        else:
            if "gender" in args.feature:
                (
                    male_val_loader,
                    female_val_loader,
                    male_test_loader,
                    female_test_loader,
                ) = load_dataloaders(args, logger, tokenizer, args.template_id)

                logger.log("start evaluation")
                logger.log("compute male data score")
                evaluate_one_dataloder(
                    args, logger, male_test_loader, prefix_model, model, tokenizer
                )
                logger.log("compute female data score")
                evaluate_one_dataloder(
                    args, logger, female_test_loader, prefix_model, model, tokenizer
                )

            elif "age" in args.feature:
                (
                    one_val_loader,
                    two_val_loader,
                    three_val_loader,
                    four_val_loader,
                    five_val_loader,
                    six_val_loader,
                    seven_val_loader,
                    one_test_loader,
                    two_test_loader,
                    three_test_loader,
                    four_test_loader,
                    five_test_loader,
                    six_test_loader,
                    seven_test_loader,
                ) = load_dataloaders(args, logger, tokenizer, args.template_id)

                logger.log("start evaluation")
                """
                logger.log("compute one data score")
                evaluate_one_dataloder(one_test_loader, prefix_model, model, tokenizer)
                logger.log("compute two data score")
                evaluate_one_dataloder(two_test_loader, prefix_model, model, tokenizer)
                logger.log("compute three data score")
                evaluate_one_dataloder(
                    three_test_loader, prefix_model, model, tokenizer
                )
                logger.log("compute four data score")
                evaluate_one_dataloder(four_test_loader, prefix_model, model, tokenizer)
                logger.log("compute five data score")
                evaluate_one_dataloder(five_test_loader, prefix_model, model, tokenizer)
                """
                logger.log("compute six data score")
                evaluate_one_dataloder(
                    args, logger, six_test_loader, prefix_model, model, tokenizer
                )
                logger.log("compute seven data score")
                evaluate_one_dataloder(
                    args, logger, seven_test_loader, prefix_model, model, tokenizer
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
        elif args.combine_method == "attention":
            multi_prompt = CFunctionPrompt(
                args=args,
                prefix_prompt_models=[prefix_model_gender, prefix_model_age],
                preseqlength=args.prefix_length,
            ).cuda()
            multi_prompt.load_state_dict(torch.load(args.prefix_pretrained_dir))

        logger.log("load data")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        if not args.compute_parity:
            val_loader, test_loader = load_dataloaders(
                args, logger, tokenizer, args.template_id
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
            ) = load_dataloaders(args, logger, tokenizer, args.template_id)

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
    parser.add_argument("--task", type=str, default="movie")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gpu", type=str, default="7")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--distributed", action="store_true")

    parser.add_argument("--prefix_pretrained_dir", type=str)
    parser.add_argument("--prefix_length", type=int, default=5)
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

    # movieevaldataset parameters
    parser.add_argument("--movie_category_negative_sample", type=int, default=5)
    parser.add_argument("--negative_sample", type=int, default=5)
    parser.add_argument("--sequential_num", type=int, default=25)
    parser.add_argument("--yes_no_sample", type=int, default=5)
    parser.add_argument("--max_history", type=int, default=20)

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
