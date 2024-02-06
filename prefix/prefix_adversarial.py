import torch
from prefix_tuning import (
    PrefixTuningT5,
    AttentionTuningT5,
    ConcatPrompt,
    AveragePrompt,
    MixingPrompt,
    AttnFFNPrefixTuningT5,
)
import sys
import argparse
import time
from datetime import datetime, date


sys.path.append("..")

from discriminator import BinaryDiscriminator, MultiDiscriminator
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5Config
from discriminator_dataset import load_dataloaders, load_insurance_dataloaders

import transformers
from transformers import T5ForConditionalGenerationwithPrefix
from modeling_p5 import P5
from tqdm import tqdm
from sklearn import metrics
import json
import random
import numpy as np
import os
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from collections import OrderedDict

from prefix_insurance_evaluate_all import (
    evaluate_and_save as insurance_evaluate_and_save,
)
from prefix_direct_insurance_evaluate_all import (
    evaluate_and_save as direct_insurance_evaluate_and_save,
)
from prefix_evaluate_all import evaluate_and_save as movie_evaluate_and_save

transformers.logging.set_verbosity_error()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


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
                today = date.today()
                today_date = today.strftime("%d/%m/%Y")
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                string = today_date + ", " + current_time + ": " + string
                logf.write(string)
                if newline:
                    logf.write("\n")

            sys.stdout.write(string)
            if newline:
                sys.stdout.write("\n")
            sys.stdout.flush()


def create_optimizer_and_scheduler(args, model, model_name, train_loader):
    # https://github.com/google-research/bert/blob/master/optimization.py#L25
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
    if model_name == "p5":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.p5_lr, eps=args.adam_eps
        )
    else:
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.dis_lr, eps=args.adam_eps
        )

    num_train_steps = int(
        len(train_loader)
        / args.gradient_accumulation_steps
        * args.initial_discriminator_epoch
    )
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
    )

    return optimizer, scheduler


def load_model(model, pretrained_dir):
    ckpt = torch.load(pretrained_dir, map_location="cuda:0")
    new_model = OrderedDict()
    for n, p in ckpt.items():
        n = n.replace("module.", "")
        new_model[n] = p
    model.load_state_dict(new_model, strict=False)

    return model


def discriminator_training(args, logger, log=True):
    args.discriminator_training = True
    logger.log("initialize model ...")
    config = T5Config.from_pretrained("t5-small")
    config.initialization = "normal"  # "zero" if args.task == "movie" else "normal"
    model = T5ForConditionalGenerationwithPrefix.from_pretrained("t5-small").cuda()

    if not args.combine_prompts:
        logger.log("load trained prefix model for a single prefix prompt model...")
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
        if args.prefix_model_dir is None:
            args.prefix_model_dir = args.unbiased_model_dir
        prefix_model.load_state_dict(torch.load(args.prefix_model_dir))
    else:
        logger.log(
            "load trained prefix model for a concatenated prefix prompt model..."
        )
        prefix_model_gender = AttentionTuningT5(
            config, preseqlen=args.gender_prefix_length
        ).cuda()
        prefix_model_age = AttentionTuningT5(
            config, preseqlen=args.age_prefix_length
        ).cuda()

        assert args.gender_prefix_pretrained_dir is not None
        assert args.age_prefix_pretrained_dir is not None

        prefix_model_gender.load_state_dict(
            torch.load(args.gender_prefix_pretrained_dir)
        )
        prefix_model_age.load_state_dict(torch.load(args.age_prefix_pretrained_dir))

        if args.combine_method == "concatenation":
            prefix_model = ConcatPrompt([prefix_model_gender, prefix_model_age])
        elif args.combine_method == "average":
            prefix_model = AveragePrompt([prefix_model_gender, prefix_model_age])
        elif args.combine_method == "mixing":
            prefix_model = MixingPrompt(
                [prefix_model_gender, prefix_model_age], args.index_chosen
            )

    logger.log("load pretrained P5 model ...")
    model = load_model(model, args.P5_pretrained_dir)

    logger.log("initialize discriminator model ...")
    if "gender" in args.feature:
        discriminator = BinaryDiscriminator(args)
    else:
        discriminator = MultiDiscriminator(args)
    discriminator.apply(discriminator.init_weights)
    discriminator = discriminator.cuda()

    # if os.path.isfile(args.feature + "_discriminator.pt"):
    #    logger.log("model trained, load pretrained model")
    #    discriminator.load_state_dict(torch.load(args.feature + "_discriminator.pt"))

    logger.log("loading data ...")
    if args.task == "movie":
        train_loader, test_loader = load_dataloaders(args)
    else:
        assert args.task == "insurance"
        train_loader, test_loader = load_insurance_dataloaders(args)

    if log:
        logger.log("creating optimizer and scheduler for discriminator")
    discriminator_optimizer, discriminator_scheduler = create_optimizer_and_scheduler(
        args, discriminator, "discriminator", train_loader
    )

    logger.log("start training")
    prefix_model.zero_grad()
    prefix_model.eval()
    model.zero_grad()
    discriminator.zero_grad()
    best_accuracy = 0
    step_num = 0
    total_loss = 0
    auc = 0
    for e in range(args.initial_discriminator_epoch):
        discriminator.train()
        for batch in tqdm(train_loader):
            input_ids = batch[0].cuda()
            attention_mask = batch[1].cuda()
            whole_word_ids = batch[2].cuda()
            output_ids = batch[3].cuda()
            user_ids = batch[5].cuda()
            discriminator_label = batch[6].cuda()

            # B * length * embedding_dim
            out = prefix_model(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                whole_word_ids=whole_word_ids,
                labels=output_ids,
                return_dict=True,
            )
            # B * length * embedding_dim
            embeddings = out["encoder_last_hidden_state"]
            B = embeddings.size(0)
            user_embeddings = []
            for b in range(B):
                user_embedding = embeddings[b][user_ids[b][0] : user_ids[b][1]].mean(
                    dim=0
                )
                user_embeddings.append(user_embedding.unsqueeze(0))
            # B * embedding_dim
            user_embeddings = torch.cat(user_embeddings, dim=0).cuda()

            loss = discriminator(user_embeddings.detach(), discriminator_label)
            total_loss += loss.item()
            step_num += 1
            if log:
                if step_num % args.discriminator_logging_step == 0:
                    logger.log(
                        "discriminator loss after {} steps is {}".format(
                            step_num, total_loss
                        )
                    )
                    total_loss = 0

            # update only the recommendation model
            loss.backward()
            discriminator_optimizer.step()
            discriminator_scheduler.step()
            discriminator.zero_grad()

        discriminator.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            total = 0
            correct = 0
            for batch in tqdm(test_loader):
                input_ids = batch[0].cuda()
                attention_mask = batch[1].cuda()
                whole_word_ids = batch[2].cuda()
                output_ids = batch[3].cuda()
                user_ids = batch[5].cuda()
                discriminator_label = batch[6].cuda()

                # B * length * embedding_dim
                out = prefix_model(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    whole_word_ids=whole_word_ids,
                    decoder_input_ids=output_ids,
                    return_dict=True,
                )
                # B * length * embedding_dim
                embeddings = out["encoder_last_hidden_state"]
                B = embeddings.size(0)
                user_embeddings = []
                for b in range(B):
                    user_embedding = embeddings[b][
                        user_ids[b][0] : user_ids[b][1]
                    ].mean(dim=0)
                    user_embeddings.append(user_embedding.unsqueeze(0))
                # B * embedding_dim
                user_embeddings = torch.cat(user_embeddings, dim=0).cuda()

                prediction = (
                    discriminator.predict(user_embeddings.detach())["prediction"]
                    .int()
                    .squeeze()
                )
                total += prediction.numel()
                correct += (prediction == discriminator_label).sum().tolist()

                predictions += (
                    discriminator.predict(user_embeddings.detach())["output"]
                    .squeeze()
                    .tolist()
                )
                labels += discriminator_label.squeeze().tolist()
        accuracy = correct / total
        logger.log("accuracy is {} after {} epochs".format(accuracy, e))
        if "gender" in args.feature:
            fpr, tpr, _ = metrics.roc_curve(
                np.array(labels), np.array(predictions), pos_label=1
            )
            auc = metrics.auc(fpr, tpr)
            logger.log("AUC score is {} after {} epochs".format(auc, e))
        else:
            macro_roc_auc_ovo = roc_auc_score(
                np.array(labels),
                np.array(predictions),
                multi_class="ovo",
                average="macro",
            )
            weighted_roc_auc_ovo = roc_auc_score(
                np.array(labels),
                np.array(predictions),
                multi_class="ovo",
                average="weighted",
            )
            macro_roc_auc_ovr = roc_auc_score(
                np.array(labels),
                np.array(predictions),
                multi_class="ovr",
                average="macro",
            )
            weighted_roc_auc_ovr = roc_auc_score(
                np.array(labels),
                np.array(predictions),
                multi_class="ovr",
                average="weighted",
            )
            logger.log(
                "One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
                "(weighted by prevalence)".format(
                    macro_roc_auc_ovo, weighted_roc_auc_ovo
                )
            )
            logger.log(
                "One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
                "(weighted by prevalence)".format(
                    macro_roc_auc_ovr, weighted_roc_auc_ovr
                )
            )
            auc = macro_roc_auc_ovo
    args.discriminator_training = False
    return auc


def adversarial_training(args, logger):
    logger.log("loading pretrained P5 model ...")
    config = T5Config.from_pretrained(args.model_type)
    model = T5ForConditionalGenerationwithPrefix.from_pretrained("t5-small").cuda()
    logger.log("load train P5 model")
    model = load_model(model, args.P5_pretrained_dir)
    config.initialization = (
        "zero" if args.task == "movie" and ("age" in args.feature or "occupation" in args.feature) else "normal"
    )

    if args.use_attention:
        prefix_model = AttentionTuningT5(
            config=config,
            preseqlen=args.prefix_length,
            attnseqlen=args.attn_prefix_length,
        ).cuda()
    else:
        logger.log("initialize prefix prompt model")
        prefix_model = PrefixTuningT5(
            config=config, preseqlen=args.prefix_length
        ).cuda()

    if args.use_trained_initialization:
        logger.log(
            "initialize prefix model from dir: {}".format(args.initialized_prefix_dir)
        )
        if os.path.exists(args.initialized_prefix_dir):
            prefix_model.load_state_dict(
                torch.load(args.initialized_prefix_dir), strict=False
            )

    if args.load_previously_trained:
        logger.log(
            "load trained unbiased model from dir: {}".format(args.unbiased_model_dir)
        )
        if os.path.exists(args.unbiased_model_dir):
            prefix_model.load_state_dict(torch.load(args.unbiased_model_dir))

    logger.log("building discriminator model ...")
    if args.task == "movie":
        if "gender" in args.feature:
            discriminator = BinaryDiscriminator(args)
            if not args.from_scratch:
                logger.log("loading trained gender discriminator model ...")
                disc_pretrained_dir = "../adversarial/user_gender_discriminator.pt"
                discriminator = load_model(discriminator, disc_pretrained_dir)
        else:
            discriminator = MultiDiscriminator(args)
            if not args.from_scratch:
                if "age" in args.feature:
                    disc_pretrained_dir = "../adversarial/user_age_discriminator.pt"
                    discriminator = load_model(discriminator, disc_pretrained_dir)
                else:
                    assert "occupation" in args.feature
                    disc_pretrained_dir = (
                        "../adversarial/user_occupation_discriminator.pt"
                    )
                    discriminator = load_model(discriminator, disc_pretrained_dir)
    if args.task == "insurance":
        discriminator = MultiDiscriminator(args)
        if not args.from_scratch:
            if "age" in args.feature:
                disc_pretrained_dir = (
                    "../adversarial/insurance/user_age_discriminator.pt"
                )
                discriminator = load_model(discriminator, disc_pretrained_dir)
            elif "occupation" in args.feature:
                assert "occupation" in args.feature
                disc_pretrained_dir = (
                    "../adversarial/insurance/user_occupation_discriminator.pt"
                )
                discriminator = load_model(discriminator, disc_pretrained_dir)
            else:
                assert "marital" in args.feature
                disc_pretrained_dir = (
                    "../adversarial/insurance/user_occupation_discriminator.pt"
                )
                discriminator = load_model(discriminator, disc_pretrained_dir)
    discriminator = discriminator.cuda()

    logger.log("loading data ...")
    if args.task == "movie":
        train_loader, test_loader = load_dataloaders(args)
    elif args.task == "insurance":
        train_loader, test_loader = load_insurance_dataloaders(args)

    if args.freeze_partial:
        logger.log("creating optimizer and scheduler for prefix model")
        optimizer = AdamW(
            prefix_model.attention_module.parameters(),
            lr=args.prefix_lr,
            eps=args.adam_eps,
        )
    else:
        optimizer = AdamW(
            prefix_model.parameters(), lr=args.prefix_lr, eps=args.adam_eps,
        )
    logger.log("creating optimizer and scheduler for discriminator")
    discriminator_optimizer = AdamW(
        discriminator.parameters(), lr=args.dis_lr, eps=args.adam_eps
    )

    logger.log("start training")
    prefix_model.zero_grad()
    discriminator.zero_grad()
    num_steps = 0
    num_tested = 0

    # though not necessary to set to False, but
    for j, p in enumerate(model.parameters()):
        p.requires_grad_(False)

    if args.use_attention:
        times = 20
    else:
        times = 10
    for e in range(args.adversarial_epoch):

        logger.log("****** adversarial prefix tuning epoch {} ******".format(e))
        for batch_num, batch in enumerate(tqdm(train_loader)):
            prefix_model.train()
            input_ids = batch[0].cuda()
            attention_mask = batch[1].cuda()
            whole_word_ids = batch[2].cuda()
            output_ids = batch[3].cuda()
            output_attention = batch[4].cuda()
            user_ids = batch[5].cuda()
            discriminator_label = batch[6].cuda()

            num_tested += 1

            for p in discriminator.parameters():
                p.requires_grad_(False)
            # B * length * embedding_dim
            for _ in range(times):
                out = prefix_model.adversarial(
                    model=model,
                    discriminator=discriminator,
                    discriminator_label=discriminator_label,
                    input_ids=input_ids,
                    feature_boundary_ids=user_ids,
                    discriminator_weight=args.discriminator_weight,
                    attention_mask=attention_mask,
                    whole_word_ids=whole_word_ids,
                    labels=output_ids,
                    labels_attention=output_attention,
                    return_dict=True,
                    train_discriminator=False,
                )
                loss = out["loss"]
                rec_loss = out["rec_loss"]
                discriminator_loss = out["discriminator_loss"]
                user_embeddings = out["feature_embeddings"]

                if num_steps % args.adversarial_logging_steps == 0:
                    print("{}".format(loss))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(prefix_model.parameters(), args.clip)
                optimizer.step()
                prefix_model.zero_grad()

            if num_steps % args.adversarial_logging_steps == 0:
                logger.log(
                    "after {} steps of adversarial, fixing discriminator, rec_loss {} - discriminator_loss {} is {}".format(
                        num_steps, rec_loss, discriminator_loss, loss
                    )
                )

            if (batch_num + 1) % args.rs_step == 0:
                if num_steps % args.adversarial_logging_steps == 0:
                    logger.log("rec los --------------- discriminator loss")
                for j, p in enumerate(discriminator.parameters()):
                    p.requires_grad_(True)
                # update the discriminator and prefix_model together for k steps
                # B * length * embedding_dim
                for i in range(args.together_discriminator_update_steps):
                    out = prefix_model.adversarial(
                        model=model,
                        discriminator=discriminator,
                        discriminator_label=discriminator_label,
                        input_ids=input_ids,
                        feature_boundary_ids=user_ids,
                        discriminator_weight=args.discriminator_weight,
                        attention_mask=attention_mask,
                        whole_word_ids=whole_word_ids,
                        labels=output_ids,
                        labels_attention=output_attention,
                        return_hidden_state=True,
                        return_dict=True,
                        train_discriminator=True,
                    )
                    loss = out["loss"]
                    rec_loss = out["rec_loss"]
                    discriminator_loss = out["discriminator_loss"]
                    user_embeddings = out["feature_embeddings"]

                    # loss.backward()
                    if i < args.rec_update_steps:
                        rec_loss.backward()
                        optimizer.step()
                        prefix_model.zero_grad()

                        out = prefix_model.adversarial(
                            model=model,
                            discriminator=discriminator,
                            discriminator_label=discriminator_label,
                            input_ids=input_ids,
                            feature_boundary_ids=user_ids,
                            discriminator_weight=args.discriminator_weight,
                            attention_mask=attention_mask,
                            whole_word_ids=whole_word_ids,
                            labels=output_ids,
                            labels_attention=output_attention,
                            return_hidden_state=True,
                            return_dict=True,
                            train_discriminator=True,
                        )
                        discriminator_loss = out["discriminator_loss"]
                        user_embeddings = out["feature_embeddings"]
                        discriminator_loss.backward()
                        discriminator_optimizer.step()
                        discriminator.zero_grad()

                    else:
                        discriminator_loss.backward()
                        discriminator_optimizer.step()
                        discriminator.zero_grad()

                    if num_steps % args.adversarial_logging_steps == 0:
                        print(str((rec_loss.item(), discriminator_loss.item())))

                for _ in range(args.sole_discriminator_update_steps):
                    discriminator_loss = discriminator(
                        user_embeddings.detach(), discriminator_label
                    )
                    discriminator_loss.backward()
                    discriminator_optimizer.step()
                    discriminator.zero_grad()
                    if num_steps % args.adversarial_logging_steps == 0:
                        print(str(discriminator_loss.item()))

                num_steps += 1

            if (
                args.task == "movie"
                and (num_steps + 1) % args.save_steps == 0
                and (num_tested + 1) % (args.save_steps * args.rs_step) == 0
            ):
                logger.log(
                    "****** save prefix prompt model to {} after epoch {}******".format(
                        args.unbiased_model_dir, e
                    )
                )
                torch.save(prefix_model.state_dict(), args.unbiased_model_dir)
                logger.log(
                    "****** evaluate adversarial model after {} steps ******".format(
                        num_steps + 1
                    )
                )
                discriminator_training(args, logger)
                args.prefix_pretrained_dir = args.unbiased_model_dir
                train_batch_size = args.batch_size
                args.batch_size = args.evaluate_batch_size
                movie_evaluate_and_save(args, logger)
                args.batch_size = train_batch_size

        logger.log(
            "****** save prefix prompt model to {} after epoch {}******".format(
                args.unbiased_model_dir, e
            )
        )
        torch.save(prefix_model.state_dict(), args.unbiased_model_dir)
        logger.log(
            "****** evaluate adversarial model after {} steps ******".format(
                num_steps + 1
            )
        )
        auc = discriminator_training(args, logger)
        if auc >= 0.55:
            torch.save(
                prefix_model.state_dict(), "pretrained_" + args.unbiased_model_dir
            )
        prefix_model.eval()
        args.prefix_pretrained_dir = args.unbiased_model_dir
        train_batch_size = args.batch_size
        args.batch_size = args.evaluate_batch_size
        if args.task == "insurance":
            if args.insurance_type == "sequential":
                insurance_evaluate_and_save(args, logger)
            else:
                direct_insurance_evaluate_and_save(args, logger)
        else:
            movie_evaluate_and_save(args, logger)
        args.batch_size = train_batch_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2022)

    # directory
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument(
        "--task", type=str, default="movie", help="movie, insurance, AliEC"
    )
    parser.add_argument("--feature", type=str, default="user_gender")
    parser.add_argument("--logging_dir", type=str, default="adversarial.log")
    parser.add_argument("--toy", action="store_true")

    # model training parameters
    parser.add_argument("--model_type", type=str, default="t5-small")
    parser.add_argument(
        "--unbiased_model_dir", type=str, default="unbiased_prefix_model.pt"
    )
    parser.add_argument(
        "--P5_pretrained_dir", type=str, default="../pretrain_movie_t5-small.pt"
    )
    parser.add_argument("--prefix_model_dir", type=str)
    parser.add_argument("--prefix_pretrained_dir", type=str)
    parser.add_argument("--prefix_length", type=int, default=5)
    parser.add_argument("--attn_prefix_length", type=int, default=5)
    parser.add_argument("--gender_prefix_length", type=int, default=5)
    parser.add_argument("--age_prefix_length", type=int, default=1)
    parser.add_argument("--clip", type=float, default=1)

    # optimizer hyperparameter
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--discriminator_training", type=bool, default=False)
    parser.add_argument("--discriminator_batch_size", type=int, default=16)
    parser.add_argument("--prefix_lr", type=float, default=1e-3)
    parser.add_argument("--dis_lr", type=float, default=1e-4)
    parser.add_argument("--adam_eps", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # adversarial hyperparameters
    parser.add_argument("--initial_discriminator_epoch", type=int, default=2)
    parser.add_argument("--adversarial_epoch", type=int, default=2)
    parser.add_argument("--discriminator_epoch", type=int, default=2)
    parser.add_argument("--discriminator_logging_step", type=int, default=500)
    parser.add_argument("--adversarial_logging_steps", type=int, default=500)
    parser.add_argument("--discriminator_weight", type=float, default=1)
    parser.add_argument("--sole_discriminator_update_steps", type=int, default=10)
    parser.add_argument("--together_discriminator_update_steps", type=int, default=10)
    parser.add_argument("--rec_update_steps", type=int, default=10)

    # evaluate on combination
    parser.add_argument(
        "--combine_prompts",
        action="store_true",
        help="use in discriminator AUC evaluation time",
    )
    parser.add_argument(
        "--combine_method",
        type=str,
        default="concatenation",
        help="differet methods: concatenation, average, mixing",
    )
    parser.add_argument("--index_chosen", nargs="+", type=int)
    parser.add_argument("--gender_prefix_pretrained_dir", type=str)
    parser.add_argument("--age_prefix_pretrained_dir", type=str)

    parser.add_argument(
        "--rs_step",
        type=int,
        default=1,
        help="number of steps that update recommendation system before update discriminator",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10000,
        help="recommendation model to save for every x steps",
    )

    # used on load_data function
    parser.add_argument(
        "--selected_train_data",
        type=str,
        default="../adversarial/selected_train_dataset.json",
    )
    parser.add_argument(
        "--selected_test_data",
        type=str,
        default="../adversarial/selected_test_dataset.json",
    )
    parser.add_argument(
        "--occupation_train_data",
        type=str,
        default="../adversarial/occupation_train_dataset.json",
    )
    parser.add_argument(
        "--occupation_test_data",
        type=str,
        default="../adversarial/occupation_test_dataset.json",
    )
    parser.add_argument(
        "--train_data", type=str, default="../adversarial/train_dataset.json"
    )
    parser.add_argument(
        "--test_data", type=str, default="../adversarial/test_dataset.json"
    )
    parser.add_argument(
        "--insurance_train_data",
        type=str,
        default="../data/insurance/insurance_train_data.json",
    )
    parser.add_argument(
        "--insurance_test_data",
        type=str,
        default="../data/insurance/insurance_test_data.json",
    )
    parser.add_argument(
        "--insurance_occupation_train_data",
        type=str,
        default="../data/insurance/occupation_train.json",
    )
    parser.add_argument(
        "--insurance_occupation_test_data",
        type=str,
        default="../data/insurance/occupation_test.json",
    )
    parser.add_argument(
        "--insurance_marital_train_data",
        type=str,
        default="../data/insurance/marital_train.json",
    )
    parser.add_argument(
        "--insurance_marital_test_data",
        type=str,
        default="../data/insurance/marital_test.json",
    )
    parser.add_argument(
        "--direct_insurance_marital_train_data",
        type=str,
        default="../data/insurance/direct_marital_train.json",
    )
    parser.add_argument(
        "--direct_insurance_marital_test_data",
        type=str,
        default="../data/insurance/direct_marital_test.json",
    )
    parser.add_argument("--insurance_type", type=str, default="sequential")

    # setting
    parser.add_argument("--train_initial_discriminator", action="store_true")
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--load_previously_trained", action="store_true")
    parser.add_argument("--use_trained_initialization", action="store_true")
    parser.add_argument("--freeze_partial", action="store_true")
    parser.add_argument("--use_attention", action="store_true")

    parser.add_argument(
        "--initialized_prefix_dir",
        type=str,
        default="trained_prefix/insurance/attentiontuningp5_direct_marital_unbiased_prefix_model_reconly.pt",
    )

    # gpu/cpu
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--distributed", action="store_true")

    # evaluation
    parser.add_argument("--compute_parity", action="store_true")
    parser.add_argument("--template_id", type=str, default="2-1")
    parser.add_argument("--evaluate_batch_size", type=int, default=1)
    parser.add_argument("--use_item_representation", action="store_true")
    # movieevaldataset parameters
    parser.add_argument("--movie_category_negative_sample", type=int, default=5)
    parser.add_argument("--negative_sample", type=int, default=5)
    parser.add_argument("--sequential_num", type=int, default=25)
    parser.add_argument("--yes_no_sample", type=int, default=5)
    parser.add_argument("--max_history", type=int, default=20)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_seed(args)
    logger = Logger(args.logging_dir, True)
    logger.log(str(args))

    if args.train_initial_discriminator:
        discriminator_training(args, logger)
    else:
        adversarial_training(args, logger)
