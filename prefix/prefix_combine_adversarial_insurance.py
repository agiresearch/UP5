import torch
from prefix_tuning import (
    PrefixTuningT5,
    ConcatPrompt,
    AveragePrompt,
    MixingPrompt,
    CFunctionPrompt,
)
import sys
import argparse

sys.path.append("..")

from discriminator import BinaryDiscriminator, MultiDiscriminator
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5Config
from discriminator_combine_dataset import load_insurance_dataloaders
from discriminator_dataset import (
    load_insurance_dataloaders as load_insurance_dataloaders_in_discriminator,
)

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

transformers.logging.set_verbosity_error()


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
    logger.log("initialize model ...")
    config = T5Config.from_pretrained("t5-small")
    config.initialization = "zero" if args.task == "movie" else "normal"
    model = T5ForConditionalGenerationwithPrefix.from_pretrained("t5-small").cuda()

    if args.attention_method:
        logger.log("load trained prefix model for an Attn prefix prompt model...")
        prefix_model_age = AttnPrefixTuningT5(
            config, preseqlen=args.age_prefix_length
        ).cuda()
        prefix_model_marital = AttnPrefixTuningT5(
            config, preseqlen=args.marital_prefix_length
        ).cuda()
        prefix_model_occupation = AttnPrefixTuningT5(
            config, preseqlen=args.occupation_prefix_length
        ).cuda()
    else:
        logger.log("load trained prefix model for an FFN prefix prompt model...")
        prefix_model_age = PrefixTuningT5(
            config, preseqlen=args.age_prefix_length
        ).cuda()
        prefix_model_marital = PrefixTuningT5(
            config, preseqlen=args.marital_prefix_length
        ).cuda()
        prefix_model_occupation = PrefixTuningT5(
            config, preseqlen=args.occupation_prefix_length
        ).cuda()

    if args.train_initial_discriminator:
        prefix_model_age.load_state_dict(torch.load(args.age_prefix_pretrained_dir))
        prefix_model_marital.load_state_dict(
            torch.load(args.marital_prefix_pretrained_dir)
        )
        prefix_model_occupation.load_state_dict(
            torch.load(args.occupation_prefix_pretrained_dir)
        )
    else:
        prefix_model_age.load_state_dict(
            torch.load("combine_" + args.age_prefix_pretrained_dir)
        )
        prefix_model_marital.load_state_dict(
            torch.load("combine_" + args.marital_prefix_pretrained_dir)
        )
        prefix_model_occupation.load_state_dict(
            torch.load("combine_" + args.occupation_prefix_pretrained_dir)
        )

    prefix_model = CFunctionPrompt(
        args,
        [prefix_model_age, prefix_model_marital, prefix_model_occupation],
        preseqlength=args.prefix_length,
    ).cuda()
    prefix_model.load_state_dict(torch.load(args.unbiased_model_dir))

    logger.log("load pretrained P5 model ...")
    model = load_model(model, args.pretrained_dir)

    logger.log("initialize discriminator model ...")

    age_discriminator = MultiDiscriminator(args, labels=5)
    marital_discriminator = MultiDiscriminator(args, labels=3)
    occupation_discriminator = MultiDiscriminator(args, labels=3)

    age_discriminator.apply(age_discriminator.init_weights)
    marital_discriminator.apply(marital_discriminator.init_weights)
    occupation_discriminator.apply(occupation_discriminator.init_weights)

    age_discriminator = age_discriminator.cuda()
    marital_discriminator = marital_discriminator.cuda()
    occupation_discriminator = occupation_discriminator.cuda()

    # if os.path.isfile(args.feature + "_discriminator.pt"):
    #    logger.log("model trained, load pretrained model")
    #    discriminator.load_state_dict(torch.load(args.feature + "_discriminator.pt"))

    logger.log("loading data ...")
    train_loader, test_loader = load_insurance_dataloaders(args, batch_size=2)

    logger.log("creating optimizer and scheduler for discriminator")
    age_optimizer, age_scheduler = create_optimizer_and_scheduler(
        args, age_discriminator, "discriminator", train_loader
    )
    marital_optimizer, marital_scheduler = create_optimizer_and_scheduler(
        args, marital_discriminator, "discriminator", train_loader
    )
    occupation_optimizer, occupation_scheduler = create_optimizer_and_scheduler(
        args, occupation_discriminator, "discriminator", train_loader
    )

    logger.log("start training")

    prefix_model.zero_grad()
    model.zero_grad()
    age_optimizer.zero_grad()
    marital_optimizer.zero_grad()
    occupation_optimizer.zero_grad()

    age_step_num = 0
    age_total_loss = 0
    marital_step_num = 0
    marital_total_loss = 0
    occupation_step_num = 0
    occupation_total_loss = 0
    for _ in range(args.initial_discriminator_epoch):
        age_discriminator.train()
        marital_discriminator.train()
        occupation_discriminator.train()
        for batch in train_loader:
            input_ids = batch[0].cuda()
            attention_mask = batch[1].cuda()
            whole_word_ids = batch[2].cuda()
            output_ids = batch[3].cuda()
            user_ids = batch[5].cuda()
            age_label = batch[6].cuda()
            marital_label = batch[7].cuda()
            occupation_label = batch[8].cuda()
            random_feature = batch[9].cuda()
            random_feature[-1] = 0
            if sum(random_feature) == 0:
                continue

            # B * length * embedding_dim
            out = prefix_model(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                whole_word_ids=whole_word_ids,
                labels=output_ids,
                return_dict=True,
                model_indices=random_feature.tolist(),
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

            ### update age ###
            age_loss = age_discriminator(user_embeddings.detach(), age_label)
            age_total_loss += age_loss.item()
            age_step_num += 1
            if log:
                if age_step_num % args.discriminator_logging_step == 0:
                    logger.log(
                        "discriminator loss after {} steps is {}".format(
                            age_step_num, age_total_loss
                        )
                    )
                    age_total_loss = 0

                if age_step_num % 10000 == 0:
                    torch.save(
                        age_discriminator.state_dict(),
                        "adversarial/combine/age_discriminator.pt",
                    )

            age_loss.backward()
            age_optimizer.step()
            age_scheduler.step()
            age_discriminator.zero_grad()

            ### update marital ###
            if not -1 in marital_label.tolist():
                marital_loss = marital_discriminator(
                    user_embeddings.detach(), marital_label
                )
                marital_total_loss += marital_loss.item()
                marital_step_num += 1
                if log:
                    if marital_step_num % args.discriminator_logging_step == 0:
                        logger.log(
                            "discriminator loss after {} steps is {}".format(
                                marital_step_num, marital_total_loss
                            )
                        )
                        marital_total_loss = 0

                    if marital_step_num % 10000 == 0:
                        torch.save(
                            marital_discriminator.state_dict(),
                            "adversarial/combine/marital_discriminator.pt",
                        )

                marital_loss.backward()
                marital_optimizer.step()
                marital_scheduler.step()
                marital_discriminator.zero_grad()

            ### update occupation ###
            if not -1 in occupation_label.tolist():
                occupation_loss = occupation_discriminator(
                    user_embeddings.detach(), occupation_label
                )
                occupation_total_loss += occupation_loss.item()
                occupation_step_num += 1
                if log:
                    if occupation_step_num % args.discriminator_logging_step == 0:
                        logger.log(
                            "discriminator loss after {} steps is {}".format(
                                occupation_step_num, occupation_total_loss
                            )
                        )
                        occupation_total_loss = 0

                    if occupation_step_num % 10000 == 0:
                        torch.save(
                            occupation_discriminator.state_dict(),
                            "adversarial/combine/occupation_discriminator.pt",
                        )

                # update only the recommendation model
                occupation_loss.backward()
                occupation_optimizer.step()
                occupation_scheduler.step()
                occupation_discriminator.zero_grad()

        age_discriminator.eval()
        age_predictions = []
        age_labels = []
        marital_discriminator.eval()
        marital_predictions = []
        marital_labels = []
        occupation_discriminator.eval()
        occupation_predictions = []
        occupation_labels = []
        with torch.no_grad():
            age_total = 0
            age_correct = 0
            marital_total = 0
            marital_correct = 0
            occupation_total = 0
            occupation_correct = 0
            for batch in tqdm(test_loader):
                input_ids = batch[0].cuda()
                attention_mask = batch[1].cuda()
                whole_word_ids = batch[2].cuda()
                output_ids = batch[3].cuda()
                user_ids = batch[5].cuda()
                age_label = batch[6].cuda()
                marital_label = batch[7].cuda()
                occupation_label = batch[8].cuda()
                random_feature = batch[9].cuda()

                # B * length * embedding_dim
                out = prefix_model(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    whole_word_ids=whole_word_ids,
                    decoder_input_ids=output_ids,
                    return_dict=True,
                    model_indices=random_feature.tolist(),
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

                # age result
                age_prediction = (
                    age_discriminator.predict(user_embeddings.detach())["prediction"]
                    .int()
                    .squeeze()
                )
                age_total += age_prediction.numel()
                age_correct += (age_prediction == age_label).sum().tolist()
                age_predictions += (
                    age_discriminator.predict(user_embeddings.detach())["output"]
                    .squeeze()
                    .tolist()
                )
                age_labels += age_label.squeeze().tolist()

                # marital result
                if -1 not in marital_label.tolist():
                    marital_prediction = (
                        marital_discriminator.predict(user_embeddings.detach())[
                            "prediction"
                        ]
                        .int()
                        .squeeze()
                    )
                    marital_total += marital_prediction.numel()
                    marital_correct += (
                        (marital_prediction == marital_label).sum().tolist()
                    )
                    marital_predictions += (
                        marital_discriminator.predict(user_embeddings.detach())[
                            "output"
                        ]
                        .squeeze()
                        .tolist()
                    )
                    marital_labels += marital_label.squeeze().tolist()

                # occupation result
                if -1 not in occupation_label.tolist():
                    occupation_prediction = (
                        age_discriminator.predict(user_embeddings.detach())[
                            "prediction"
                        ]
                        .int()
                        .squeeze()
                    )
                    occupation_total += occupation_prediction.numel()
                    occupation_correct += (
                        (occupation_prediction == occupation_label).sum().tolist()
                    )
                    occupation_predictions += (
                        occupation_discriminator.predict(user_embeddings.detach())[
                            "output"
                        ]
                        .squeeze()
                        .tolist()
                    )
                    occupation_labels += occupation_label.squeeze().tolist()

        # AUC score
        #### age ####
        macro_roc_auc_ovo = roc_auc_score(
            np.array(age_labels),
            np.array(age_predictions),
            multi_class="ovo",
            average="macro",
        )
        weighted_roc_auc_ovo = roc_auc_score(
            np.array(age_labels),
            np.array(age_predictions),
            multi_class="ovo",
            average="weighted",
        )
        macro_roc_auc_ovr = roc_auc_score(
            np.array(age_labels),
            np.array(age_predictions),
            multi_class="ovr",
            average="macro",
        )
        weighted_roc_auc_ovr = roc_auc_score(
            np.array(age_labels),
            np.array(age_predictions),
            multi_class="ovr",
            average="weighted",
        )
        logger.log(
            "age One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
            "(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo)
        )
        logger.log(
            "age One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
            "(weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr)
        )

        #### marital ####
        macro_roc_auc_ovo = roc_auc_score(
            np.array(marital_labels),
            np.array(marital_predictions),
            multi_class="ovo",
            average="macro",
        )
        weighted_roc_auc_ovo = roc_auc_score(
            np.array(marital_labels),
            np.array(marital_predictions),
            multi_class="ovo",
            average="weighted",
        )
        macro_roc_auc_ovr = roc_auc_score(
            np.array(marital_labels),
            np.array(marital_predictions),
            multi_class="ovr",
            average="macro",
        )
        weighted_roc_auc_ovr = roc_auc_score(
            np.array(marital_labels),
            np.array(marital_predictions),
            multi_class="ovr",
            average="weighted",
        )
        logger.log(
            "marital One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
            "(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo)
        )
        logger.log(
            "marital One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
            "(weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr)
        )

        #### occupation ####
        macro_roc_auc_ovo = roc_auc_score(
            np.array(occupation_labels),
            np.array(occupation_predictions),
            multi_class="ovo",
            average="macro",
        )
        weighted_roc_auc_ovo = roc_auc_score(
            np.array(occupation_labels),
            np.array(occupation_predictions),
            multi_class="ovo",
            average="weighted",
        )
        macro_roc_auc_ovr = roc_auc_score(
            np.array(occupation_labels),
            np.array(occupation_predictions),
            multi_class="ovr",
            average="macro",
        )
        weighted_roc_auc_ovr = roc_auc_score(
            np.array(occupation_labels),
            np.array(occupation_predictions),
            multi_class="ovr",
            average="weighted",
        )
        logger.log(
            "occupation One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
            "(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo)
        )
        logger.log(
            "occupation One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
            "(weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr)
        )


def adversarial_training(args, logger):
    logger.log("loading pretrained P5 model ...")
    config = T5Config.from_pretrained(args.model_type)
    config.initialization = "zero" if args.task == "movie" else "normal"
    model = T5ForConditionalGenerationwithPrefix.from_pretrained("t5-small").cuda()
    logger.log("load train P5 model")
    model = load_model(model, args.pretrained_dir)

    if args.attention_method:
        logger.log(
            "load trained prefix model for an Attn-combined prefix prompt model..."
        )
        prefix_model_age = AttnPrefixTuningT5(
            config, preseqlen=args.age_prefix_length
        ).cuda()
        prefix_model_marital = AttnPrefixTuningT5(
            config, preseqlen=args.marital_prefix_length
        ).cuda()
        prefix_model_occupation = AttnPrefixTuningT5(
            config, preseqlen=args.occupation_prefix_length
        ).cuda()
    else:
        logger.log("load trained prefix model for an FFN prefix prompt model...")
        prefix_model_age = PrefixTuningT5(
            config, preseqlen=args.age_prefix_length
        ).cuda()
        prefix_model_marital = PrefixTuningT5(
            config, preseqlen=args.marital_prefix_length
        ).cuda()
        prefix_model_occupation = PrefixTuningT5(
            config, preseqlen=args.occupation_prefix_length
        ).cuda()

    for p in prefix_model_age.parameters():
        p.requires_grad_(False)
    for p in prefix_model_marital.parameters():
        p.requires_grad_(False)
    for p in prefix_model_occupation.parameters():
        p.requires_grad_(False)

    if not args.prefix_prompt_scratch:
        assert args.age_prefix_pretrained_dir is not None
        assert args.marital_prefix_pretrained_dir is not None
        assert args.occupation_prefix_pretrained_dir is not None
        prefix_model_age.load_state_dict(torch.load(args.age_prefix_pretrained_dir))
        prefix_model_marital.load_state_dict(
            torch.load(args.marital_prefix_pretrained_dir)
        )
        prefix_model_occupation.load_state_dict(
            torch.load(args.occupation_prefix_pretrained_dir)
        )

    if args.keep_training:
        assert os.path.isfile("upward_" + args.age_prefix_pretrained_dir)
        assert os.path.isfile("upward_" + args.marital_prefix_pretrained_dir)
        assert os.path.isfile("upward_" + args.occupation_prefix_pretrained_dir)
        prefix_model_age.load_state_dict(
            torch.load("upward_" + args.age_prefix_pretrained_dir)
        )
        prefix_model_marital.load_state_dict(
            torch.load("upward_" + args.marital_prefix_pretrained_dir)
        )
        prefix_model_occupation.load_state_dict(
            torch.load("upward_" + args.occupation_prefix_pretrained_dir)
        )

    prefix_model = CFunctionPrompt(
        args,
        [prefix_model_age, prefix_model_marital, prefix_model_occupation],
        preseqlength=args.prefix_length,
    )
    if args.keep_training:
        prefix_model.load_state_dict(torch.load(args.unbiased_model_dir))

    logger.log("building discriminator model ...")
    age_discriminator = MultiDiscriminator(args, labels=5)
    marital_discriminator = MultiDiscriminator(args, labels=3)
    occupation_discriminator = MultiDiscriminator(args, labels=3)
    if not args.from_scratch:
        logger.log("loading trained age discriminator model ...")
        age_pretrained_dir = "../adversarial/insurance/user_age_discriminator.pt"
        age_discriminator = load_model(age_discriminator, age_pretrained_dir)
        marital_pretrained_dir = (
            "../adversarial/insurance/user_marital_discriminator.pt"
        )
        marital_discriminator = load_model(
            marital_discriminator, marital_pretrained_dir
        )
        occupation_pretrained_dir = (
            "../adversarial/insurance/user_occupation_discriminator.pt"
        )
        occupation_discriminator = load_model(
            occupation_discriminator, occupation_pretrained_dir
        )
    age_discriminator = age_discriminator.cuda()
    marital_discriminator = marital_discriminator.cuda()
    occupation_discriminator = occupation_discriminator.cuda()

    logger.log("loading data ...")
    train_loader, test_loader = load_insurance_dataloaders(args)

    logger.log("creating optimizer and scheduler for prefix models")
    optimizer = AdamW(prefix_model.parameters(), lr=args.prefix_lr, eps=args.adam_eps,)
    logger.log("creating optimizer and scheduler for discriminator")
    age_discriminator_optimizer = AdamW(
        age_discriminator.parameters(), lr=args.dis_lr, eps=args.adam_eps
    )
    marital_discriminator_optimizer = AdamW(
        marital_discriminator.parameters(), lr=args.dis_lr, eps=args.adam_eps
    )
    occupation_discriminator_optimizer = AdamW(
        occupation_discriminator.parameters(), lr=args.dis_lr, eps=args.adam_eps
    )

    logger.log("start training")
    prefix_model.zero_grad()
    marital_discriminator.zero_grad()
    age_discriminator.zero_grad()
    occupation_discriminator.zero_grad()
    num_steps = 0

    # though not necessary to set to False, but
    for j, p in enumerate(model.parameters()):
        p.requires_grad_(False)

    for e in range(args.adversarial_epoch):

        logger.log("****** adversarial prefix tuning epoch {} ******".format(e))
        for batch_num, batch in enumerate(tqdm(train_loader)):
            input_ids = batch[0].cuda()
            attention_mask = batch[1].cuda()
            whole_word_ids = batch[2].cuda()
            output_ids = batch[3].cuda()
            output_attention = batch[4].cuda()
            user_ids = batch[5].cuda()
            age_label = batch[6].cuda()
            marital_label = batch[7].cuda()
            occupation_label = batch[8].cuda()
            random_feature = batch[9].cuda()  # for every batch

            if sum(random_feature) == 1:
                continue
            if -1 in age_label or -1 in marital_label or -1 in occupation_label:
                continue

            for p in age_discriminator.parameters():
                p.requires_grad_(False)
            for p in marital_discriminator.parameters():
                p.requires_grad_(False)
            for p in occupation_discriminator.parameters():
                p.requires_grad_(False)
            # B * length * embedding_dim
            for _ in range(20):
                out = prefix_model.combine_adversarial(
                    model=model,
                    discriminators=[
                        age_discriminator,
                        marital_discriminator,
                        occupation_discriminator,
                    ],
                    age_label=age_label,
                    marital_label=marital_label,
                    occupation_label=occupation_label,
                    labels=output_ids,
                    labels_attention=output_attention,
                    discriminator_feature=random_feature,
                    input_ids=input_ids,
                    feature_boundary_ids=user_ids,
                    age_weight=args.age_weight,
                    marital_weight=args.marital_weight,
                    occupation_weight=args.occupation_weight,
                    attention_mask=attention_mask,
                    whole_word_ids=whole_word_ids,
                    return_dict=True,
                    train_discriminator=False,
                )
                loss = out["loss"]
                rec_loss = out["rec_loss"]
                discriminator_loss = out["discriminator_loss"]
                user_embeddings = out["feature_embeddings"]

                if num_steps % args.adversarial_logging_steps == 0:
                    logger.log("{}".format(loss))

                loss.backward()
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
                for p in age_discriminator.parameters():
                    p.requires_grad_(True)
                for p in marital_discriminator.parameters():
                    p.requires_grad_(True)
                for p in occupation_discriminator.parameters():
                    p.requires_grad_(True)
                # update the discriminator and prefix_model together for k steps
                # B * length * embedding_dim
                for i in range(args.together_discriminator_update_steps):
                    out = prefix_model.combine_adversarial(
                        model=model,
                        discriminators=[
                            age_discriminator,
                            marital_discriminator,
                            occupation_discriminator,
                        ],
                        age_label=age_label,
                        marital_label=marital_label,
                        occupation_label=occupation_label,
                        discriminator_feature=random_feature,
                        input_ids=input_ids,
                        feature_boundary_ids=user_ids,
                        age_weight=args.age_weight,
                        marital_weight=args.marital_weight,
                        occupation_weight=args.occupation_weight,
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
                    user_embeddings = out["feature_embeddings"]

                    if i < args.rec_update_steps:
                        rec_loss.backward()
                        optimizer.step()
                        prefix_model.zero_grad()

                        if random_feature[0]:
                            out = prefix_model.combine_adversarial(
                                model=model,
                                discriminators=[
                                    age_discriminator,
                                    marital_discriminator,
                                    occupation_discriminator,
                                ],
                                age_label=age_label,
                                marital_label=marital_label,
                                occupation_label=occupation_label,
                                discriminator_feature=random_feature,
                                input_ids=input_ids,
                                feature_boundary_ids=user_ids,
                                age_weight=args.age_weight,
                                marital_weight=args.marital_weight,
                                occupation_weight=args.occupation_weight,
                                attention_mask=attention_mask,
                                whole_word_ids=whole_word_ids,
                                labels=output_ids,
                                labels_attention=output_attention,
                                return_hidden_state=True,
                                return_dict=True,
                                train_discriminator=True,
                            )
                            age_discriminator_loss = out["age_loss"]
                            age_discriminator_loss.backward()
                            age_discriminator_optimizer.step()
                            age_discriminator_optimizer.zero_grad()
                        if random_feature[1]:
                            out = prefix_model.combine_adversarial(
                                model=model,
                                discriminators=[
                                    age_discriminator,
                                    marital_discriminator,
                                    occupation_discriminator,
                                ],
                                age_label=age_label,
                                marital_label=marital_label,
                                occupation_label=occupation_label,
                                discriminator_feature=random_feature,
                                input_ids=input_ids,
                                feature_boundary_ids=user_ids,
                                age_weight=args.age_weight,
                                marital_weight=args.marital_weight,
                                occupation_weight=args.occupation_weight,
                                attention_mask=attention_mask,
                                whole_word_ids=whole_word_ids,
                                labels=output_ids,
                                labels_attention=output_attention,
                                return_hidden_state=True,
                                return_dict=True,
                                train_discriminator=True,
                            )
                            marital_discriminator_loss = out["marital_loss"]
                            marital_discriminator_loss.backward()
                            marital_discriminator_optimizer.step()
                            marital_discriminator_optimizer.zero_grad()
                        if random_feature[2]:
                            out = prefix_model.combine_adversarial(
                                model=model,
                                discriminators=[
                                    age_discriminator,
                                    marital_discriminator,
                                    occupation_discriminator,
                                ],
                                age_label=age_label,
                                marital_label=marital_label,
                                occupation_label=occupation_label,
                                discriminator_feature=random_feature,
                                input_ids=input_ids,
                                feature_boundary_ids=user_ids,
                                age_weight=args.age_weight,
                                marital_weight=args.marital_weight,
                                occupation_weight=args.occupation_weight,
                                attention_mask=attention_mask,
                                whole_word_ids=whole_word_ids,
                                labels=output_ids,
                                labels_attention=output_attention,
                                return_hidden_state=True,
                                return_dict=True,
                                train_discriminator=True,
                            )
                            occupation_discriminator_loss = out["occupation_loss"]
                            occupation_discriminator_loss.backward()
                            occupation_discriminator_optimizer.step()
                            occupation_discriminator_optimizer.zero_grad()
                    else:
                        if random_feature[0]:
                            age_discriminator_loss = out["age_loss"]
                            age_discriminator_loss.backward()
                            age_discriminator_optimizer.step()
                            age_discriminator_optimizer.zero_grad()
                            if random_feature[1]:
                                out = prefix_model.combine_adversarial(
                                    model=model,
                                    discriminators=[
                                        age_discriminator,
                                        marital_discriminator,
                                        occupation_discriminator,
                                    ],
                                    age_label=age_label,
                                    marital_label=marital_label,
                                    occupation_label=occupation_label,
                                    discriminator_feature=random_feature,
                                    input_ids=input_ids,
                                    feature_boundary_ids=user_ids,
                                    age_weight=args.age_weight,
                                    marital_weight=args.marital_weight,
                                    occupation_weight=args.occupation_weight,
                                    attention_mask=attention_mask,
                                    whole_word_ids=whole_word_ids,
                                    labels=output_ids,
                                    labels_attention=output_attention,
                                    return_hidden_state=True,
                                    return_dict=True,
                                    train_discriminator=True,
                                )
                                marital_discriminator_loss = out["marital_loss"]
                                marital_discriminator_loss.backward()
                                marital_discriminator_optimizer.step()
                                marital_discriminator_optimizer.zero_grad()
                            if random_feature[2]:
                                out = prefix_model.combine_adversarial(
                                    model=model,
                                    discriminators=[
                                        age_discriminator,
                                        marital_discriminator,
                                        occupation_discriminator,
                                    ],
                                    age_label=age_label,
                                    marital_label=marital_label,
                                    occupation_label=occupation_label,
                                    discriminator_feature=random_feature,
                                    input_ids=input_ids,
                                    feature_boundary_ids=user_ids,
                                    age_weight=args.age_weight,
                                    marital_weight=args.marital_weight,
                                    occupation_weight=args.occupation_weight,
                                    attention_mask=attention_mask,
                                    whole_word_ids=whole_word_ids,
                                    labels=output_ids,
                                    labels_attention=output_attention,
                                    return_hidden_state=True,
                                    return_dict=True,
                                    train_discriminator=True,
                                )
                                occupation_discriminator_loss = out["occupation_loss"]
                                occupation_discriminator_loss.backward()
                                occupation_discriminator_optimizer.step()
                                occupation_discriminator_optimizer.zero_grad()
                        elif random_feature[1]:
                            marital_discriminator_loss = out["marital_loss"]
                            marital_discriminator_loss.backward()
                            marital_discriminator_optimizer.step()
                            marital_discriminator_optimizer.zero_grad()
                            if random_feature[0]:
                                out = prefix_model.combine_adversarial(
                                    model=model,
                                    discriminators=[
                                        age_discriminator,
                                        marital_discriminator,
                                        occupation_discriminator,
                                    ],
                                    age_label=age_label,
                                    marital_label=marital_label,
                                    occupation_label=occupation_label,
                                    discriminator_feature=random_feature,
                                    input_ids=input_ids,
                                    feature_boundary_ids=user_ids,
                                    age_weight=args.age_weight,
                                    marital_weight=args.marital_weight,
                                    occupation_weight=args.occupation_weight,
                                    attention_mask=attention_mask,
                                    whole_word_ids=whole_word_ids,
                                    labels=output_ids,
                                    labels_attention=output_attention,
                                    return_hidden_state=True,
                                    return_dict=True,
                                    train_discriminator=True,
                                )
                                age_discriminator_loss = out["age_loss"]
                                age_discriminator_loss.backward()
                                age_discriminator_optimizer.step()
                                age_discriminator_optimizer.zero_grad()
                            if random_feature[2]:
                                out = prefix_model.combine_adversarial(
                                    model=model,
                                    discriminators=[
                                        age_discriminator,
                                        marital_discriminator,
                                        occupation_discriminator,
                                    ],
                                    age_label=age_label,
                                    marital_label=marital_label,
                                    occupation_label=occupation_label,
                                    discriminator_feature=random_feature,
                                    input_ids=input_ids,
                                    feature_boundary_ids=user_ids,
                                    age_weight=args.age_weight,
                                    marital_weight=args.marital_weight,
                                    occupation_weight=args.occupation_weight,
                                    attention_mask=attention_mask,
                                    whole_word_ids=whole_word_ids,
                                    labels=output_ids,
                                    labels_attention=output_attention,
                                    return_hidden_state=True,
                                    return_dict=True,
                                    train_discriminator=True,
                                )
                                occupation_discriminator_loss = out["occupation_loss"]
                                occupation_discriminator_loss.backward()
                                occupation_discriminator_optimizer.step()
                                occupation_discriminator_optimizer.zero_grad()
                        elif random_feature[2]:
                            occupation_discriminator_loss = out["occupation_loss"]
                            occupation_discriminator_loss.backward()
                            occupation_discriminator_optimizer.step()
                            occupation_discriminator_optimizer.zero_grad()
                            if random_feature[0]:
                                out = prefix_model.combine_adversarial(
                                    model=model,
                                    discriminators=[
                                        age_discriminator,
                                        marital_discriminator,
                                        occupation_discriminator,
                                    ],
                                    age_label=age_label,
                                    marital_label=marital_label,
                                    occupation_label=occupation_label,
                                    discriminator_feature=random_feature,
                                    input_ids=input_ids,
                                    feature_boundary_ids=user_ids,
                                    age_weight=args.age_weight,
                                    marital_weight=args.marital_weight,
                                    occupation_weight=args.occupation_weight,
                                    attention_mask=attention_mask,
                                    whole_word_ids=whole_word_ids,
                                    labels=output_ids,
                                    labels_attention=output_attention,
                                    return_hidden_state=True,
                                    return_dict=True,
                                    train_discriminator=True,
                                )
                                age_discriminator_loss = out["age_loss"]
                                age_discriminator_loss.backward()
                                age_discriminator_optimizer.step()
                                age_discriminator_optimizer.zero_grad()
                            if random_feature[1]:
                                out = prefix_model.combine_adversarial(
                                    model=model,
                                    discriminators=[
                                        age_discriminator,
                                        marital_discriminator,
                                        occupation_discriminator,
                                    ],
                                    age_label=age_label,
                                    marital_label=marital_label,
                                    occupation_label=occupation_label,
                                    discriminator_feature=random_feature,
                                    input_ids=input_ids,
                                    feature_boundary_ids=user_ids,
                                    age_weight=args.age_weight,
                                    marital_weight=args.marital_weight,
                                    occupation_weight=args.occupation_weight,
                                    attention_mask=attention_mask,
                                    whole_word_ids=whole_word_ids,
                                    labels=output_ids,
                                    labels_attention=output_attention,
                                    return_hidden_state=True,
                                    return_dict=True,
                                    train_discriminator=True,
                                )
                                marital_discriminator_loss = out["marital_loss"]
                                marital_discriminator_loss.backward()
                                marital_discriminator_optimizer.step()
                                marital_discriminator_optimizer.zero_grad()

                for _ in range(args.sole_discriminator_update_steps):
                    if random_feature[0]:
                        age_discriminator_loss = age_discriminator(
                            user_embeddings.detach(), age_label
                        )
                        age_discriminator_loss.backward()
                        age_discriminator_optimizer.step()
                        age_discriminator.zero_grad()
                        if num_steps % args.adversarial_logging_steps == 0:
                            logger.log(str(age_discriminator_loss.item()))
                    if random_feature[1]:
                        marital_discriminator_loss = marital_discriminator(
                            user_embeddings.detach(), marital_label
                        )
                        marital_discriminator_loss.backward()
                        marital_discriminator_optimizer.step()
                        marital_discriminator.zero_grad()
                        if num_steps % args.adversarial_logging_steps == 0:
                            logger.log(str(marital_discriminator_loss.item()))
                    if random_feature[2]:
                        occupation_discriminator_loss = occupation_discriminator(
                            user_embeddings.detach(), occupation_label
                        )
                        occupation_discriminator_loss.backward()
                        occupation_discriminator_optimizer.step()
                        occupation_discriminator.zero_grad()
                        if num_steps % args.adversarial_logging_steps == 0:
                            logger.log(str(occupation_discriminator_loss.item()))

                num_steps += 1

            if (num_steps + 1) % args.save_steps == 0:
                torch.save(prefix_model.state_dict(), args.unbiased_model_dir)
                torch.save(
                    prefix_model_age.state_dict(),
                    "upward_" + args.age_prefix_pretrained_dir,
                )
                torch.save(
                    prefix_model_marital.state_dict(),
                    "upward_" + args.marital_prefix_pretrained_dir,
                )
                torch.save(
                    prefix_model_occupation.state_dict(),
                    "upward_" + args.occupation_prefix_pretrained_dir,
                )

        logger.log(
            "****** evaluate adversarial model after epoch {} ******".format(e + 1)
        )
        discriminator_training(args, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)

    ##### directory
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument(
        "--task", type=str, default="movie", help="movie, insurance, AliEC"
    )

    ##### insurance data directory
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
    parser.add_argument("--insurance_type", type=str, default="sequential")

    ##### other training directory
    parser.add_argument("--feature", type=str, default="user_gender")
    parser.add_argument("--logging_dir", type=str, default="adversarial.log")
    parser.add_argument("--toy", action="store_true")

    # model training parameters
    parser.add_argument("--model_type", type=str, default="t5-small")
    parser.add_argument(
        "--unbiased_model_dir", type=str, default="unbiased_prefix_model.pt"
    )
    parser.add_argument(
        "--pretrained_dir", type=str, default="../pretrain_movie_t5-small.pt"
    )
    parser.add_argument("--prefix_model_dir", type=str)
    parser.add_argument("--prefix_length", type=int, default=5)

    parser.add_argument("--age_prefix_length", type=int, default=1)
    parser.add_argument("--marital_prefix_length", type=int, default=5)
    parser.add_argument("--occupation_prefix_length", type=int, default=5)

    # optimizer hyperparameter
    parser.add_argument("--batch_size", type=int, default=16)
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
    parser.add_argument("--discriminator_logging_step", type=int, default=500)
    parser.add_argument("--adversarial_logging_steps", type=int, default=500)

    parser.add_argument("--age_weight", type=float, default=1)
    parser.add_argument("--marital_weight", type=float, default=1)
    parser.add_argument("--occupation_weight", type=float, default=1)

    parser.add_argument("--sole_discriminator_update_steps", type=int, default=10)
    parser.add_argument("--together_discriminator_update_steps", type=int, default=10)
    parser.add_argument("--rec_update_steps", type=int, default=10)

    # evaluate on combination
    parser.add_argument("--age_prefix_pretrained_dir", type=str)
    parser.add_argument("--marital_prefix_pretrained_dir", type=str)
    parser.add_argument("--occupation_prefix_pretrained_dir", type=str)

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

    parser.add_argument("--train_initial_discriminator", action="store_true")

    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--prefix_prompt_scratch", action="store_true")
    parser.add_argument("--keep_training", action="store_true")
    parser.add_argument("--attention_method", action="store_true")

    # gpu/cpu
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--distributed", action="store_true")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_seed(args)
    logger = Logger(args.logging_dir, True)
    logger.log(str(args))

    if args.train_initial_discriminator:
        discriminator_training(args, logger)
    else:
        adversarial_training(args, logger)
