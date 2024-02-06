import transformers

transformers.logging.set_verbosity_error()

import torch
import argparse

from dataset import load_dataloaders
from transformers import T5Tokenizer, T5Config
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from modeling_p5 import P5
import random
import numpy as np
import os
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import sys
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import collections
import json
from collections import OrderedDict


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


def create_optimizer_and_scheduler(args, logger, model, train_loader):
    batch_per_epoch = len(train_loader)
    total_steps = batch_per_epoch // args.gradient_accumulation_steps * args.epochs
    warmup_steps = int(total_steps * args.warmup_prop)

    if args.gpu == 0:
        logger.log("Batch per epoch: %d" % batch_per_epoch)
        logger.log("Total steps: %d" % total_steps)
        logger.log("Warmup proportion:", args.warmup_prop)
        logger.log("Warm up steps: %d" % warmup_steps)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    return optimizer, scheduler


def exact_match(predictions, targets):
    correct = 0
    for p, t in zip(predictions, targets):
        if p == t:
            correct += 1

    return correct


def load_model(model, pretrained_dir, rank):
    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    ckpt = torch.load(pretrained_dir, map_location=map_location)
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        k = k.replace("module_", "")
        new_ckpt[k] = v

    model.load_state_dict(new_ckpt)

    return model


def trainer(args, rank, train_loader, val_loader, tokenizer, logger):
    if rank == 0:
        logger.log("loading model ...")
    config = T5Config.from_pretrained(args.model_type)
    model = P5.from_pretrained(args.model_type, config=config).to(args.gpu)
    # model = P5(config).to(args.gpu)
    # if rank == 0:
    #    logger.log("finished building model")
    # if os.path.isfile(args.model_dir):
    #    logger.log("load pretrained model")
    # configure map_location properly
    #    model = load_model(model, "good" + args.model_dir, rank)

    optimizer, scheduler = create_optimizer_and_scheduler(
        args, logger, model, train_loader
    )

    if args.distributed:
        dist.barrier()

    if args.multiGPU:
        if rank == 0:
            logger.log("model dataparallel set")
        if args.distributed:
            model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)

    if rank == 0:
        logger.log("start training")
    model.zero_grad()
    logging_step = 0
    logging_loss = 0
    best_precision = 0.0
    for e in range(args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(e)
        model.train()
        for batch in train_loader:
            input_ids = batch[0].to(args.gpu)
            attn = batch[1].to(args.gpu)
            whole_input_ids = batch[2].to(args.gpu)
            output_ids = batch[3].to(args.gpu)
            output_attention = batch[4].to(args.gpu)

            if args.distributed:
                output = model.module(
                    input_ids=input_ids,
                    whole_word_ids=whole_input_ids,
                    attention_mask=attn,
                    labels=output_ids,
                    alpha=args.alpha,
                    return_dict=True,
                )
            else:
                output = model(
                    input_ids=input_ids,
                    whole_word_ids=whole_input_ids,
                    attention_mask=attn,
                    labels=output_ids,
                    alpha=args.alpha,
                    return_dict=True,
                )

            # compute loss masking padded tokens
            loss = output["loss"]
            lm_mask = output_attention != 0
            lm_mask = lm_mask.float()
            B, L = output_ids.size()
            loss = loss.view(B, L) * lm_mask
            loss = (loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)).mean()

            logging_loss += loss.item()

            # update
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            logging_step += 1

            if logging_step % args.logging_step == 0 and rank == 0:
                logger.log(
                    "total loss for {} steps : {}".format(logging_step, logging_loss)
                )
                logging_loss = 0

        dist.barrier()

        if rank == 0:
            logger.log("start evaluation after epoch {}".format(e))
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0].to(args.gpu)
                attn = batch[1].to(args.gpu)
                whole_input_ids = batch[2].to(args.gpu)
                output_ids = batch[3].to(args.gpu)
                output_attention = batch[4].to(args.gpu)

                if args.distributed:
                    prediction = model.module.generate(
                        input_ids=input_ids, attention_mask=attn, max_length=8
                    )
                else:
                    prediction = model.generate(
                        input_ids=input_ids, attention_mask=attn, max_length=8
                    )

                gold_sents = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                generated_sents = tokenizer.batch_decode(
                    prediction, skip_special_tokens=True
                )

                correct += exact_match(generated_sents, gold_sents)
                total += len(gold_sents)

        result = [correct, total]
        with open(
            "result/{}_val_result_{}_{}.json".format(args.task, rank, e), "w"
        ) as f:
            json.dump(result, f)
        all_computed = True
        for i in range(4):
            if not os.path.isfile(
                "result/{}_val_result_{}_{}.json".format(args.task, i, e)
            ):
                all_computed = False
        if all_computed:
            c, t = 0, 0
            for i in range(args.world_size):
                with open(
                    "result/{}_val_result_{}_{}.json".format(args.task, i, e), "r"
                ) as f:
                    one_precision = json.load(f)
                    c += one_precision[0]
                    t += one_precision[1]
            precision = c / t

            logger.log("exact match precision is {} for epoch {}".format(precision, e))
            if precision > best_precision:
                if args.further_train:
                    model_dir = "further" + args.model_dir
                else:
                    model_dir = args.model_dir
                if args.distributed:
                    torch.save(model.module.state_dict(), model_dir)
                else:
                    torch.save(model.state_dict(), model_dir)
                best_precision = precision

        dist.barrier()


def main_worker(local_rank, args, logger):
    args.gpu = local_rank
    args.rank = local_rank
    logger.log(f"Process Launching at GPU {args.gpu}")

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(
            backend="nccl", world_size=args.world_size, rank=args.rank
        )

    logger.log(f"Building train loader at GPU {args.gpu}")

    if local_rank == 0:
        logger.log("loading data ...")
    tokenizer = T5Tokenizer.from_pretrained(args.model_type)
    train_loader, val_loader, test_loader = load_dataloaders(args, tokenizer)
    if local_rank == 0:
        logger.log("finished loading data")
        logger.log("length of training data is {}".format(len(train_loader)))
        logger.log("length of val data is {}".format(len(val_loader)))
        logger.log("length of test data is {}".format(len(test_loader)))

    trainer(args, local_rank, train_loader, val_loader, tokenizer, logger)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)

    # directory
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument(
        "--task", type=str, default="movie", help="movie, insurance, AliEC"
    )
    parser.add_argument(
        "--insurance_type", type=str, help="only for insurance, whether direct or not"
    )
    parser.add_argument("--logging_dir", type=str)
    parser.add_argument("--further_train", action="store_true")

    # data generation in data_preprocessing_movie.py
    parser.add_argument("--movie_category_negative_sample", type=int, default=5)
    parser.add_argument("--negative_sample", type=int, default=5)
    parser.add_argument("--sequential_num", type=int, default=25)
    parser.add_argument("--yes_no_sample", type=int, default=5)
    parser.add_argument("--max_history", type=int, default=20)
    parser.add_argument("--direct_item_proportion", type=int, default=2)

    # model type
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--model_type", type=str, default="t5-base")

    # training parameter
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, default=1)
    parser.add_argument("--logging_step", type=int, default=100)
    parser.add_argument("--warmup_prop", type=float, default=0.05)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_eps", type=float, default=1e-6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=2)

    # CPU/GPU
    parser.add_argument("--multiGPU", action="store_const", default=False, const=True)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--gpu", type=str, default="1,2,3")
    parser.add_argument("--local_rank", type=int, default=-1)

    # data
    parser.add_argument("--use_item_representation", action="store_true")
    parser.add_argument(
        "--item_representation",
        type=str,
        default="random_vocab_mapping",
        help="hash_table, random_vocab_mapping, vocab_mapping",
    )

    args = parser.parse_args()

    if not args.use_item_representation:
        if args.insurance_type != "direct":
            args.model_dir = "pretrain_" + args.task + "_{}.pt".format(args.model_type)
        else:
            args.model_dir = (
                "pretrain_direct_" + args.task + "_{}.pt".format(args.model_type)
            )
    else:
        args.model_dir = (
            "{}/pretrain_".format(args.item_representation)
            + args.task
            + "_{}.pt".format(args.model_type)
        )
    if args.toy:
        args.model_dir = "toy_" + args.model_dir

    return args


if __name__ == "__main__":
    transformers.logging.set_verbosity_error()

    cudnn.benchmark = True
    args = parse_argument()

    set_seed(args)
    logger = Logger(args.logging_dir, True)
    logger.log(str(args))

    # number of visible gpus set in os[environ]
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    if args.distributed:
        mp.spawn(
            main_worker, args=(args, logger), nprocs=args.world_size, join=True,
        )
