import torch
from transformers import (
    T5Tokenizer,
    T5Config,
    T5ForConditionalGenerationwithPrefix,
    T5PreTrainedModelwithPrefix,
)
from torch import nn
from torch.nn import CrossEntropyLoss
import time
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
import random
import numpy as np
import transformers
import os
from typing import Optional, Tuple, Union, List

transformers.logging.set_verbosity_error()


def set_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

##### Combine method #####
class CFunctionPrompt(nn.Module):
    def __init__(self, args, prefix_prompt_models, preseqlength=5):
        super().__init__()
        self.args = args
        self.prefix_prompt_models = prefix_prompt_models
        self.preseqlen = preseqlength
        self.query_embedding = nn.Embedding(self.preseqlen * 9 * 4, 512).cuda()
        self.query = nn.Linear(512, 512).cuda()
        self.key = nn.Linear(512, 512).cuda()
        self.value = nn.Linear(512, 512).cuda()
        self.combination_function = nn.MultiheadAttention(
            512, 8, batch_first=True
        ).cuda()

        nn.init.xavier_normal_(self.query.weight)
        nn.init.xavier_normal_(self.key.weight)
        nn.init.xavier_normal_(self.value.weight)

    def combine_promts(self, input_ids, model_indices):
        bsz = input_ids.shape[0]

        past_key_values_prompts = []

        if sum(model_indices) == 1:
            past_key_values_prompt = self.prefix_prompt_models[
                model_indices.index(1)
            ].get_prompt(bsz=bsz)
            return past_key_values_prompt

        for prefix_prompt_model in [
            self.prefix_prompt_models[i] for i, v in enumerate(model_indices) if v == 1
        ]:
            past_key_values_prompt = prefix_prompt_model.get_prompt(bsz=bsz)
            past_key_values_prompts.append(past_key_values_prompt)
        # past_key_values_prompt = self.prefix_prompt_models[1].get_prompt(bsz=bsz)
        # past_key_values_prompts.append(past_key_values_prompt)

        attention_combined_prompt = []
        query_id = 0
        # for every layer
        for layer in range(len(past_key_values_prompts[0])):
            # for each layer, all prompts
            one_layer_prompts = [
                past_key_values_prompt[layer]
                for past_key_values_prompt in past_key_values_prompts
            ]

            one_layer_combined_prompt = []
            for j in range(len(one_layer_prompts[0])):
                if one_layer_prompts[0][j].dim() == 4:
                    one_dim, two_dim, three_dim, four_dim = one_layer_prompts[0][
                        j
                    ].size()
                    one_concatenated_prompt = torch.cat(
                        [one_layer_prompt[j] for one_layer_prompt in one_layer_prompts],
                        dim=-2,
                    )
                    one_concatenated_prompt = one_concatenated_prompt.transpose(
                        2, 1
                    ).reshape(bsz, self.preseqlen * sum(model_indices), -1)

                    key = self.key(one_concatenated_prompt)
                    value = self.value(one_concatenated_prompt)
                    query = self.query(
                        self.query_embedding(
                            torch.tensor(
                                list(
                                    range(
                                        self.preseqlen * query_id,
                                        self.preseqlen * (query_id + 1),
                                    )
                                )
                            ).to(one_layer_prompts[0][0].device)
                        ).repeat(bsz, 1, 1)
                    )

                    combined_prompt, _ = self.combination_function(query, key, value,)
                    combined_prompt = combined_prompt.reshape(
                        one_dim, two_dim, three_dim, four_dim
                    )
                    query_id += 1
                else:
                    combined_prompt = one_layer_prompts[0][j]

                one_layer_combined_prompt.append(combined_prompt)

            attention_combined_prompt.append(one_layer_combined_prompt)

        return attention_combined_prompt

    def forward(self, model, model_indices, input_ids, **kwargs):
        concatenated_prmopts = self.combine_promts(input_ids, model_indices)

        past_prefix_prompts = concatenated_prmopts

        output = model(
            input_ids=input_ids, past_prefix_prompts=past_prefix_prompts, **kwargs,
        )

        return output

    def generate(
        self, model, model_indices, input_ids=None, **kwargs,
    ):

        concatenated_prmopts = self.combine_promts(input_ids, model_indices)

        past_prefix_prompts = concatenated_prmopts

        inputs = {
            "input_ids": input_ids,
            "past_prefix_prompts": past_prefix_prompts,
        }
        output = model.generate(**inputs, **kwargs)

        return output

    def combine_adversarial(
        self,
        model,
        input_ids=None,
        labels=None,
        labels_attention=None,
        past_key_values=None,
        discriminators=None,
        gender_label=None,
        age_label=None,
        marital_label=None,
        occupation_label=None,
        discriminator_feature=None,
        feature_boundary_ids=None,
        gender_weight=None,
        age_weight=None,
        marital_weight=None,
        occupation_weight=None,
        whole_word_ids=None,
        attention_mask=None,
        train_discriminator=False,
        return_hidden_state=True,
        reduce_loss=False,
        **kwargs,
    ):
        bsz = input_ids.shape[0]

        combined_prmopts = self.combine_promts(
            input_ids, discriminator_feature.tolist()
        )
        past_prefix_prompts = combined_prmopts

        past_prefix_prompts_encoder = [
            past_prefix_prompt[:3] for past_prefix_prompt in past_prefix_prompts
        ]

        encoder_outputs = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            whole_word_ids=whole_word_ids,
            past_prefix_prompts=past_prefix_prompts_encoder,
        )
        hidden_states = encoder_outputs[0]

        B = hidden_states.size(0)
        user_embeddings = None
        for b in range(B):
            user_embedding = (
                hidden_states[b][
                    feature_boundary_ids[b][0] : feature_boundary_ids[b][1]
                ]
                .mean(dim=0)
                .unsqueeze(0)
            )
            if b == 0:
                user_embeddings = user_embedding
            else:
                user_embeddings = torch.cat([user_embeddings, user_embedding], dim=0)

        # B * embedding_dim
        assert user_embeddings is not None
        assert discriminators is not None

        if self.args.task == "movie":
            if discriminator_feature[0] == 1:
                gender_discriminator_loss = discriminators[0](
                    user_embeddings, gender_label
                )
                upward_gender_discriminator_loss = 0
            else:
                gender_discriminator_loss = 0
                if -1 not in gender_label:
                    upward_gender_discriminator_loss = discriminators[0](
                        user_embeddings, gender_label
                    )
                else:
                    upward_gender_discriminator_loss = 0
            if discriminator_feature[1] == 1:
                age_discriminator_loss = discriminators[1](user_embeddings, age_label)
                upward_age_discriminator_loss = 0
            else:
                age_discriminator_loss = 0
                if -1 not in age_label:
                    upward_age_discriminator_loss = discriminators[0](
                        user_embeddings, gender_label
                    )
                else:
                    upward_age_discriminator_loss = 0

            discriminator_loss = (
                gender_weight * gender_discriminator_loss
                + age_weight * age_discriminator_loss
            )
            upward_discriminator_loss = (
                gender_weight * upward_gender_discriminator_loss
                + age_weight * upward_age_discriminator_loss
            )
        else:
            if discriminator_feature[0] == 1:
                age_discriminator_loss = discriminators[0](user_embeddings, age_label)
                upward_age_discriminator_loss = 0
            else:
                age_discriminator_loss = 0
                if -1 not in age_label:
                    upward_age_discriminator_loss = discriminators[0](
                        user_embeddings, age_label
                    )
                else:
                    upward_age_discriminator_loss = 0
            if discriminator_feature[1] == 1:
                marital_discriminator_loss = discriminators[1](
                    user_embeddings, marital_label
                )
                upward_marital_discriminator_loss = 0
            else:
                marital_discriminator_loss = 0
                if -1 not in marital_label:
                    upward_marital_discriminator_loss = discriminators[1](
                        user_embeddings, marital_label
                    )
                else:
                    upward_marital_discriminator_loss = 0
            if discriminator_feature[2] == 1:
                occupation_discriminator_loss = discriminators[2](
                    user_embeddings, occupation_label
                )
                upward_occupation_discriminator_loss = 0
            else:
                occupation_discriminator_loss = 0
                if -1 not in occupation_label:
                    upward_occupation_discriminator_loss = discriminators[2](
                        user_embeddings, occupation_label
                    )
                else:
                    upward_occupation_discriminator_loss = 0

            discriminator_loss = (
                age_weight * age_discriminator_loss
                + marital_weight * marital_discriminator_loss
                + occupation_weight * occupation_discriminator_loss
            )
            upward_discriminator_loss = (
                age_weight * upward_age_discriminator_loss
                + marital_weight * upward_marital_discriminator_loss
                + occupation_weight * upward_occupation_discriminator_loss
            )
            # print("discriminator_loss {}".format(discriminator_loss))
            # print("upward_discriminator_loss {}".format(upward_discriminator_loss))

        decoder_input_ids = model._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert (
                labels is None
            ), "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]

        if attention_mask is None:
            attention_mask = input_ids.ne(model.config.pad_token_id).to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
        encoder_attention_mask = attention_mask

        # Decode
        past_prefix_prompts_decoder = [
            past_prefix_prompt[3:] for past_prefix_prompt in past_prefix_prompts
        ]
        decoder_outputs = model.decoder(
            input_ids=decoder_input_ids,
            past_prefix_prompts=past_prefix_prompts_decoder,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = decoder_outputs[0]

        assert model.config.tie_word_embeddings is True

        if model.config.tie_word_embeddings:
            sequence_output = sequence_output * (model.model_dim ** -0.5)

        lm_logits = model.lm_head(sequence_output)

        rec_loss = None
        if labels is not None:
            if reduce_loss:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
            rec_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        lm_mask = labels_attention != 0
        lm_mask = lm_mask.float()
        B, L = labels.size()
        rec_loss = rec_loss.view(B, L) * lm_mask
        rec_loss = (rec_loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)).mean()

        # print("rec_loss {}".format(rec_loss))

        if not train_discriminator:
            loss = rec_loss - discriminator_loss + upward_discriminator_loss
            # loss = -discriminator_loss
            return P5AdversarialSeq2SeqLMOutput(
                loss=loss,
                rec_loss=rec_loss,
                discriminator_loss=discriminator_loss,
                upward_discriminator_loss=upward_discriminator_loss,
                feature_embeddings=user_embeddings,
            )
        else:
            loss = rec_loss + discriminator_loss
            if self.args.task == "movie":
                return P5AdversarialSeq2SeqLMOutput(
                    loss=loss,
                    rec_loss=rec_loss,
                    discriminator_loss=discriminator_loss,
                    gender_loss=gender_discriminator_loss,
                    age_loss=age_discriminator_loss,
                    feature_embeddings=user_embeddings,
                )
            else:
                return P5AdversarialSeq2SeqLMOutput(
                    loss=loss,
                    rec_loss=rec_loss,
                    discriminator_loss=discriminator_loss,
                    age_loss=age_discriminator_loss,
                    marital_loss=marital_discriminator_loss,
                    occupation_loss=occupation_discriminator_loss,
                    feature_embeddings=user_embeddings,
                )


class AttentionTuningT5(T5PreTrainedModelwithPrefix):
    def __init__(
        self, config, optim_prefix=False, preseqlen=5, attnseqlen=5,
    ):
        super().__init__(config)
        self.preseqlen = preseqlen
        self.attnseqlen = attnseqlen
        self.config = config

        self.match_n_layer = config.num_layers
        self.match_n_head = config.num_heads
        self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.input_tokens = torch.arange(self.preseqlen).long().cuda()

        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        # self.control_trans = nn.Sequential(
        #    nn.Linear(self.n_embd, 800),
        #    nn.Tanh(),
        #    nn.Linear(800, self.match_n_layer * 2 * self.n_embd),
        # )
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.match_n_layer * 2 * self.n_embd),
        )

        self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.match_n_layer * 2 * self.n_embd),
            # nn.Tanh(),
            # nn.Linear(800, 800),
            # nn.Tanh(),
            # nn.Linear(800, self.match_n_layer * 2 * self.n_embd),
        )

        self.wte_cross = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_cross = nn.Sequential(
            nn.Linear(self.n_embd, self.match_n_layer * 2 * self.n_embd),
            # nn.Tanh(),
            # nn.Linear(800, 800),
            # nn.Tanh(),
            # nn.Linear(800, self.match_n_layer * 2 * self.n_embd),
        )

        self.get_prompt = self.get_prompt_p5

        self.dropout = nn.Dropout(0.1)

        if self.config.initialization == "zero":
            self.wte.weight.data.zero_()
            self.wte_enc.weight.data.zero_()
            self.wte_cross.weight.data.zero_()
            self.control_trans[0].weight.data.zero_()
            self.control_trans[0].bias.data.zero_()
            # self.control_trans[2].weight.data.zero_()
            # self.control_trans[2].bias.data.zero_()
            self.control_trans_enc[0].weight.data.zero_()
            self.control_trans_enc[0].bias.data.zero_()
            # self.control_trans_enc[2].weight.data.zero_()
            # self.control_trans_enc[2].bias.data.zero_()
            # self.control_trans_enc[4].weight.data.zero_()
            # self.control_trans_enc[4].bias.data.zero_()
            self.control_trans_cross[0].weight.data.zero_()
            self.control_trans_cross[0].bias.data.zero_()
            # self.control_trans_cross[2].weight.data.zero_()
            # self.control_trans_cross[2].bias.data.zero_()
            # self.control_trans_cross[4].weight.data.zero_()
            # self.control_trans_cross[4].bias.data.zero_()
        else:
            nn.init.xavier_normal_(self.control_trans[0].weight)
            # nn.init.xavier_normal_(self.control_trans[2].weight)
            nn.init.xavier_normal_(self.control_trans_enc[0].weight)
            # nn.init.xavier_normal_(self.control_trans_enc[2].weight)
            # nn.init.xavier_normal_(self.control_trans_enc[4].weight)
            nn.init.xavier_normal_(self.control_trans_cross[0].weight)
            # nn.init.xavier_normal_(self.control_trans_cross[2].weight)
            # nn.init.xavier_normal_(self.control_trans_cross[4].weight)

        self.attention_module = nn.Sequential(
            nn.Embedding(self.preseqlen * 9 * 4, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 512),
            nn.MultiheadAttention(512, 8, batch_first=True),
        )

        nn.init.xavier_normal_(self.attention_module[1].weight)
        nn.init.xavier_normal_(self.attention_module[2].weight)
        nn.init.xavier_normal_(self.attention_module[3].weight)

    def get_single_prompt_p5(self, bsz=1, sample_size=1):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1)

        # encoder prefix prompt
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # decoder prefix prompt
        input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1)
        temp_control_enc = self.wte_enc(input_tokens_enc)
        past_key_values_enc = self.control_trans_enc(
            temp_control_enc
        )  # bsz, seqlen, layer*emb
        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz_enc,
            seqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        # cross prefix prompt
        input_tokens_cross = self.input_tokens.unsqueeze(0).expand(old_bsz, -1)
        temp_control_cross = self.wte_cross(input_tokens_cross)
        past_key_values_cross = self.control_trans_cross(
            temp_control_cross
        )  # bsz, seqlen, layer*emb
        bsz_enc, seqlen, _ = past_key_values_cross.shape
        past_key_values_cross = past_key_values_cross.view(
            bsz_enc,
            seqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        past_key_values_cross = self.dropout(past_key_values_cross)
        past_key_values_cross = past_key_values_cross.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            key_val_enc = past_key_values_enc[i]
            key_val_cross = past_key_values_cross[i]
            # encoder
            # decoder
            # cross attention
            temp_dict = [
                key_val[0].contiguous().cuda(),
                key_val[1].contiguous().cuda(),
                torch.zeros(bsz, seqlen).to(key_val.device).bool(),
                key_val_enc[0].contiguous().cuda(),
                key_val_enc[1].contiguous().cuda(),
                torch.zeros(bsz, seqlen).to(key_val_enc.device).bool(),
                key_val_cross[0].contiguous().cuda(),
                key_val_cross[1].contiguous().cuda(),
                torch.zeros(bsz, seqlen).to(key_val_cross.device).bool(),
            ]

            result.append(temp_dict)

        return result

    def get_prompt_p5(self, bsz=1, sample_size=1):
        past_key_values_prompt = self.get_single_prompt_p5(bsz=bsz)

        attention_combined_prompt = []

        query_id = 0

        for layer in range(len(past_key_values_prompt)):

            one_layer_prompts = past_key_values_prompt[layer]

            one_layer_combined_prompt = []

            for j in range(len(one_layer_prompts)):
                # apply attention only to encoder prompt
                if j <= 2:
                    if one_layer_prompts[j].dim() == 4:
                        one_dim, two_dim, _, four_dim = one_layer_prompts[j].size()

                        one_prompt = (
                            one_layer_prompts[j]
                            .transpose(2, 1)
                            .reshape(bsz, self.preseqlen, -1)
                        )

                        key = self.attention_module[2](one_prompt)
                        value = self.attention_module[3](one_prompt)
                        query = self.attention_module[1](
                            self.attention_module[0](
                                torch.tensor(
                                    list(
                                        range(
                                            self.attnseqlen * query_id,
                                            self.attnseqlen * (query_id + 1),
                                        )
                                    )
                                ).to(one_layer_prompts[j].device)
                            ).repeat(bsz, 1, 1)
                        )
                        combined_prompt, _ = self.attention_module[4](
                            query, key, value,
                        )
                        combined_prompt = combined_prompt.reshape(
                            one_dim, two_dim, self.attnseqlen, four_dim
                        )
                        query_id += 1
                    else:
                        combined_prompt = one_layer_prompts[j][:, : self.attnseqlen]
                else:
                    if one_layer_prompts[j].dim() == 4:
                        combined_prompt = one_layer_prompts[j]
                    else:
                        combined_prompt = one_layer_prompts[j][:, : self.attnseqlen]

                one_layer_combined_prompt.append(combined_prompt)

            attention_combined_prompt.append(one_layer_combined_prompt)

        return attention_combined_prompt

    def forward(
        self, model, input_ids=None, past_key_values=None, **kwargs,
    ):

        bsz = input_ids.shape[0]

        past_key_values_prompt = self.get_prompt(bsz=bsz)

        past_prefix_prompts = past_key_values_prompt

        output = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            past_prefix_prompts=past_prefix_prompts,
            **kwargs,
        )

        return output

    def encoder(
        self, model, input_ids=None, past_key_values=None, **kwargs,
    ):
        encoder = model.get_encoder()

        bsz = input_ids.shape[0]

        past_key_values_prompts = self.get_prompt(bsz=bsz)

        past_prefix_prompts_encoder = [
            past_key_values_prompt[:3]
            for past_key_values_prompt in past_key_values_prompts
        ]

        output = encoder(
            input_ids=input_ids,
            past_key_values=past_key_values,
            past_prefix_prompts=past_prefix_prompts_encoder,
            return_dict=True,
            **kwargs,
        )

        return output

    def generate(
        self, model, input_ids=None, **kwargs,
    ):

        bsz = input_ids.shape[0]

        past_prefix_prompts = self.get_prompt(bsz=bsz)

        inputs = {
            "input_ids": input_ids,
            "past_prefix_prompts": past_prefix_prompts,
        }
        output = model.generate(**inputs, **kwargs)

        return output

    def adversarial(
        self,
        model,
        input_ids=None,
        labels=None,
        labels_attention=None,
        past_key_values=None,
        discriminator=None,
        discriminator_label=None,
        feature_boundary_ids=None,
        discriminator_weight=None,
        whole_word_ids=None,
        attention_mask=None,
        train_discriminator=False,
        return_hidden_state=True,
        reduce_loss=False,
        **kwargs,
    ):
        bsz = input_ids.shape[0]

        past_prefix_prompts = self.get_prompt(bsz=bsz)

        past_prefix_prompts_encoder = [
            past_prefix_prompt[:3] for past_prefix_prompt in past_prefix_prompts
        ]

        encoder_outputs = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            whole_word_ids=whole_word_ids,
            past_prefix_prompts=past_prefix_prompts_encoder,
        )
        hidden_states = encoder_outputs[0]

        B = hidden_states.size(0)
        user_embeddings = None
        for b in range(B):
            user_embedding = (
                hidden_states[b][
                    feature_boundary_ids[b][0] : feature_boundary_ids[b][1]
                ]
                .mean(dim=0)
                .unsqueeze(0)
            )
            if b == 0:
                user_embeddings = user_embedding
            else:
                user_embeddings = torch.cat([user_embeddings, user_embedding], dim=0)
        # B * embedding_dim
        assert user_embeddings is not None

        discriminator_loss = discriminator(user_embeddings, discriminator_label)

        decoder_input_ids = model._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert (
                labels is None
            ), "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]

        if attention_mask is None:
            attention_mask = input_ids.ne(model.config.pad_token_id).to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
        encoder_attention_mask = attention_mask

        # Decode
        past_prefix_prompts_decoder = [
            past_prefix_prompt[3:] for past_prefix_prompt in past_prefix_prompts
        ]
        decoder_outputs = model.decoder(
            input_ids=decoder_input_ids,
            past_prefix_prompts=past_prefix_prompts_decoder,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = decoder_outputs[0]

        assert model.config.tie_word_embeddings is True

        if model.config.tie_word_embeddings:
            sequence_output = sequence_output * (model.model_dim ** -0.5)

        lm_logits = model.lm_head(sequence_output)

        rec_loss = None
        if labels is not None:
            if reduce_loss:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
            rec_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        lm_mask = labels_attention != 0
        lm_mask = lm_mask.float()
        B, L = labels.size()
        rec_loss = rec_loss.view(B, L) * lm_mask
        rec_loss = (rec_loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)).mean()

        if not train_discriminator:
            loss = rec_loss - discriminator_weight * discriminator_loss
            # loss = -discriminator_weight * discriminator_loss
        else:
            loss = rec_loss + discriminator_weight * discriminator_loss

        return P5AdversarialSeq2SeqLMOutput(
            loss=loss,
            rec_loss=rec_loss,
            discriminator_loss=discriminator_loss,
            feature_embeddings=user_embeddings,
        )


class PrefixTuningT5(T5PreTrainedModelwithPrefix):
    def __init__(
        self, config, optim_prefix=False, preseqlen=5,
    ):
        super().__init__(config)
        self.preseqlen = preseqlen
        self.config = config

        self.match_n_layer = config.num_layers
        self.match_n_head = config.num_heads
        self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.input_tokens = torch.arange(self.preseqlen).long().cuda()

        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, 800),
            nn.Tanh(),
            nn.Linear(800, self.match_n_layer * 2 * self.n_embd),
        )

        # self.control_trans = nn.Sequential(
        #    nn.Linear(self.n_embd, self.match_n_layer * 2 * self.n_embd),
        # )

        self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_enc = nn.Sequential(
           nn.Linear(self.n_embd, 800),
           nn.Tanh(),
           nn.Linear(800, 800),
           nn.Tanh(),
           nn.Linear(800, self.match_n_layer * 2 * self.n_embd),
        )

        #self.control_trans_enc = nn.Sequential(
        #    nn.Linear(self.n_embd, self.match_n_layer * 2 * self.n_embd)
        #)

        self.wte_cross = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_cross = nn.Sequential(
           nn.Linear(self.n_embd, 800),
           nn.Tanh(),
           nn.Linear(800, 800),
           nn.Tanh(),
           nn.Linear(800, self.match_n_layer * 2 * self.n_embd),
        )

        #self.control_trans_cross = nn.Sequential(
        #    nn.Linear(self.n_embd, self.match_n_layer * 2 * self.n_embd),
        #)

        self.get_prompt = self.get_prompt_p5

        self.dropout = nn.Dropout(0.1)

        if self.config.initialization == "zero":
            self.wte.weight.data.zero_()
            self.wte_enc.weight.data.zero_()
            self.wte_cross.weight.data.zero_()
            self.control_trans[0].weight.data.zero_()
            self.control_trans[0].bias.data.zero_()
            # self.control_trans[2].weight.data.zero_()
            # self.control_trans[2].bias.data.zero_()
            self.control_trans_enc[0].weight.data.zero_()
            self.control_trans_enc[0].bias.data.zero_()
            # self.control_trans_enc[2].weight.data.zero_()
            # self.control_trans_enc[2].bias.data.zero_()
            # self.control_trans_enc[4].weight.data.zero_()
            # self.control_trans_enc[4].bias.data.zero_()
            self.control_trans_cross[0].weight.data.zero_()
            self.control_trans_cross[0].bias.data.zero_()
            # self.control_trans_cross[2].weight.data.zero_()
            # self.control_trans_cross[2].bias.data.zero_()
            # self.control_trans_cross[4].weight.data.zero_()
            # self.control_trans_cross[4].bias.data.zero_()
        else:
            #self.wte_enc.weight.data.zero_()
            #self.wte_cross.weight.data.zero_()
            nn.init.xavier_normal_(self.control_trans[0].weight)
            nn.init.xavier_normal_(self.control_trans[2].weight)
            #self.control_trans_enc[0].weight.data.zero_()
            #self.control_trans_enc[0].bias.data.zero_()
            nn.init.xavier_normal_(self.control_trans_enc[0].weight)
            nn.init.xavier_normal_(self.control_trans_enc[2].weight)
            nn.init.xavier_normal_(self.control_trans_enc[4].weight)
            #self.control_trans_cross[0].weight.data.zero_()
            #self.control_trans_cross[0].bias.data.zero_()
            nn.init.xavier_normal_(self.control_trans_cross[0].weight)
            nn.init.xavier_normal_(self.control_trans_cross[2].weight)
            nn.init.xavier_normal_(self.control_trans_cross[4].weight)

    def get_prompt_p5(self, bsz=1, sample_size=1):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1)

        # encoder prefix prompt
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # decoder prefix prompt
        input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1)
        temp_control_enc = self.wte_enc(input_tokens_enc)
        past_key_values_enc = self.control_trans_enc(
            temp_control_enc
        )  # bsz, seqlen, layer*emb
        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz_enc,
            seqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        # cross prefix prompt
        input_tokens_cross = self.input_tokens.unsqueeze(0).expand(old_bsz, -1)
        temp_control_cross = self.wte_cross(input_tokens_cross)
        past_key_values_cross = self.control_trans_cross(
            temp_control_cross
        )  # bsz, seqlen, layer*emb
        bsz_enc, seqlen, _ = past_key_values_cross.shape
        past_key_values_cross = past_key_values_cross.view(
            bsz_enc,
            seqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        past_key_values_cross = self.dropout(past_key_values_cross)
        past_key_values_cross = past_key_values_cross.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            key_val_enc = past_key_values_enc[i]
            key_val_cross = past_key_values_cross[i]
            # encoder
            # decoder
            # cross attention
            temp_dict = [
                key_val[0].contiguous().cuda(),
                key_val[1].contiguous().cuda(),
                torch.zeros(bsz, seqlen).to(key_val.device).bool(),
                key_val_enc[0].contiguous().cuda(),
                key_val_enc[1].contiguous().cuda(),
                torch.zeros(bsz, seqlen).to(key_val_enc.device).bool(),
                key_val_cross[0].contiguous().cuda(),
                key_val_cross[1].contiguous().cuda(),
                torch.zeros(bsz, seqlen).to(key_val_cross.device).bool(),
            ]

            result.append(temp_dict)

        return result

    def forward(
        self, model, input_ids=None, past_key_values=None, **kwargs,
    ):

        bsz = input_ids.shape[0]

        past_key_values_prompt = self.get_prompt(bsz=bsz)

        past_prefix_prompts = past_key_values_prompt

        output = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            past_prefix_prompts=past_prefix_prompts,
            **kwargs,
        )

        return output

    def encoder(
        self, model, input_ids=None, past_key_values=None, **kwargs,
    ):
        encoder = model.get_encoder()

        bsz = input_ids.shape[0]

        past_key_values_prompts = self.get_prompt(bsz=bsz)

        past_prefix_prompts_encoder = [
            past_key_values_prompt[:3]
            for past_key_values_prompt in past_key_values_prompts
        ]

        output = encoder(
            input_ids=input_ids,
            past_key_values=past_key_values,
            past_prefix_prompts=past_prefix_prompts_encoder,
            return_dict=True,
            **kwargs,
        )

        return output

    def generate(
        self, model, input_ids=None, **kwargs,
    ):

        bsz = input_ids.shape[0]

        past_prefix_prompts = self.get_prompt(bsz=bsz)

        inputs = {
            "input_ids": input_ids,
            "past_prefix_prompts": past_prefix_prompts,
        }
        output = model.generate(**inputs, **kwargs)

        return output

    def adversarial(
        self,
        model,
        input_ids=None,
        labels=None,
        labels_attention=None,
        past_key_values=None,
        discriminator=None,
        discriminator_label=None,
        feature_boundary_ids=None,
        discriminator_weight=None,
        whole_word_ids=None,
        attention_mask=None,
        train_discriminator=False,
        return_hidden_state=True,
        reduce_loss=False,
        **kwargs,
    ):
        bsz = input_ids.shape[0]

        past_prefix_prompts = self.get_prompt(bsz=bsz)

        past_prefix_prompts_encoder = [
            past_prefix_prompt[:3] for past_prefix_prompt in past_prefix_prompts
        ]

        encoder_outputs = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            whole_word_ids=whole_word_ids,
            past_prefix_prompts=past_prefix_prompts_encoder,
        )
        hidden_states = encoder_outputs[0]

        B = hidden_states.size(0)
        user_embeddings = None
        for b in range(B):
            user_embedding = (
                hidden_states[b][
                    feature_boundary_ids[b][0] : feature_boundary_ids[b][1]
                ]
                .mean(dim=0)
                .unsqueeze(0)
            )
            if b == 0:
                user_embeddings = user_embedding
            else:
                user_embeddings = torch.cat([user_embeddings, user_embedding], dim=0)
        # B * embedding_dim
        assert user_embeddings is not None

        discriminator_loss = discriminator(user_embeddings, discriminator_label)

        decoder_input_ids = model._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert (
                labels is None
            ), "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]

        if attention_mask is None:
            attention_mask = input_ids.ne(model.config.pad_token_id).to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
        encoder_attention_mask = attention_mask

        # Decode
        past_prefix_prompts_decoder = [
            past_prefix_prompt[3:] for past_prefix_prompt in past_prefix_prompts
        ]
        decoder_outputs = model.decoder(
            input_ids=decoder_input_ids,
            past_prefix_prompts=past_prefix_prompts_decoder,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = decoder_outputs[0]

        assert model.config.tie_word_embeddings is True

        if model.config.tie_word_embeddings:
            sequence_output = sequence_output * (model.model_dim ** -0.5)

        lm_logits = model.lm_head(sequence_output)

        rec_loss = None
        if labels is not None:
            if reduce_loss:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
            rec_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        lm_mask = labels_attention != 0
        lm_mask = lm_mask.float()
        B, L = labels.size()
        rec_loss = rec_loss.view(B, L) * lm_mask
        rec_loss = (rec_loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)).mean()

        if not train_discriminator:
            loss = rec_loss - discriminator_weight * discriminator_loss

        return P5AdversarialSeq2SeqLMOutput(
            loss=loss,
            rec_loss=rec_loss,
            discriminator_loss=discriminator_loss,
            feature_embeddings=user_embeddings,
        )


class NoUseAttnPrefixTuningT5(T5PreTrainedModelwithPrefix):
    """Classification Head for  transformer encoders"""

    def __init__(
        self, config, optim_prefix=False, preseqlen=5,
    ):
        super().__init__(config)
        self.preseqlen = preseqlen
        self.config = config

        self.match_n_layer = config.num_layers
        self.match_n_head = config.num_heads
        self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.input_tokens = torch.arange(self.preseqlen).long().cuda()

        # need: self.match_n_layer * 2 * self.n_emb
        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
        self.wte_cross = nn.Embedding(self.preseqlen, self.n_embd)

        self.query_embedding = nn.Embedding(self.preseqlen * 3, self.n_embd)
        self.query = nn.Linear(self.n_embd, self.n_embd)
        """
        self.encoder_key = nn.Linear(self.n_embd, self.n_embd)
        self.encoder_value = nn.Linear(self.n_embd, self.n_embd)
        self.cross_key = nn.Linear(self.n_embd, self.n_embd)
        self.cross_value = nn.Linear(self.n_embd, self.n_embd)
        self.decoder_key = nn.Linear(self.n_embd, self.n_embd)
        self.decoder_value = nn.Linear(self.n_embd, self.n_embd)
        """
        self.key = nn.Linear(self.n_embd, self.n_embd)
        self.value = nn.Linear(self.n_embd, self.n_embd)

        self.combination_function = nn.MultiheadAttention(
            self.n_embd, self.match_n_head, batch_first=True
        )
        self.attention_map = nn.Linear(
            self.n_embd, self.match_n_layer * 2 * self.n_embd
        )

        nn.init.xavier_normal_(self.query.weight)
        nn.init.xavier_normal_(self.key.weight)
        nn.init.xavier_normal_(self.value.weight)
        """
        nn.init.xavier_normal_(self.encoder_key.weight)
        nn.init.xavier_normal_(self.encoder_value.weight)
        nn.init.xavier_normal_(self.decoder_key.weight)
        nn.init.xavier_normal_(self.decoder_value.weight)
        nn.init.xavier_normal_(self.cross_key.weight)
        nn.init.xavier_normal_(self.cross_value.weight)
        """

        self.get_prompt = self.get_prompt_p5

        self.dropout = nn.Dropout(0.1)

    def get_prompt_p5(self, bsz=1, sample_size=1):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1)

        ######## encoder prefix prompt ########
        encoder_embedding = self.wte(input_tokens)
        encoder_key = self.key(encoder_embedding)
        encoder_value = self.value(encoder_embedding)
        encoder_query_embedding = self.query_embedding(input_tokens)
        encoder_query = self.query(encoder_query_embedding)
        past_key_values = self.attention_map(
            self.combination_function(encoder_query, encoder_key, encoder_value)[0]
        )  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        ######## decoder prefix prompt ########
        decoder_embedding = self.wte_enc(input_tokens)
        decoder_key = self.key(decoder_embedding)
        decoder_value = self.value(decoder_embedding)
        decoder_query_embedding = self.query_embedding(input_tokens + self.preseqlen)
        decoder_query = self.query(decoder_query_embedding)
        past_key_values_enc = self.attention_map(
            self.combination_function(decoder_query, decoder_key, decoder_value)[0]
        )  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        ######## cross prefix prompt ########
        cross_embedding = self.wte_cross(input_tokens)
        cross_key = self.key(cross_embedding)
        cross_value = self.value(cross_embedding)
        cross_query_embedding = self.query_embedding(input_tokens + 2 * self.preseqlen)
        cross_query = self.query(cross_query_embedding)
        past_key_values_cross = self.attention_map(
            self.combination_function(cross_query, cross_key, cross_value)[0]
        )  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values_cross.shape
        past_key_values_cross = past_key_values_cross.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_cross = self.dropout(past_key_values_cross)
        past_key_values_cross = past_key_values_cross.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            key_val_enc = past_key_values_enc[i]
            key_val_cross = past_key_values_cross[i]
            # encoder
            # decoder
            # cross attention
            temp_dict = [
                key_val[0].contiguous().cuda(),
                key_val[1].contiguous().cuda(),
                torch.zeros(bsz, seqlen).to(key_val.device).bool(),
                key_val_enc[0].contiguous().cuda(),
                key_val_enc[1].contiguous().cuda(),
                torch.zeros(bsz, seqlen).to(key_val_enc.device).bool(),
                key_val_cross[0].contiguous().cuda(),
                key_val_cross[1].contiguous().cuda(),
                torch.zeros(bsz, seqlen).to(key_val_cross.device).bool(),
            ]

            result.append(temp_dict)

        return result

    def forward(
        self, model, input_ids=None, past_key_values=None, **kwargs,
    ):

        bsz = input_ids.shape[0]

        past_key_values_prompt = self.get_prompt(bsz=bsz)

        past_prefix_prompts = past_key_values_prompt

        output = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            past_prefix_prompts=past_prefix_prompts,
            **kwargs,
        )

        return output

    def encoder(
        self, model, input_ids=None, past_key_values=None, **kwargs,
    ):
        encoder = model.get_encoder()

        bsz = input_ids.shape[0]

        past_key_values_prompts = self.get_prompt(bsz=bsz)

        past_prefix_prompts_encoder = [
            past_key_values_prompt[:3]
            for past_key_values_prompt in past_key_values_prompts
        ]

        output = encoder(
            input_ids=input_ids,
            past_key_values=past_key_values,
            past_prefix_prompts=past_prefix_prompts_encoder,
            return_dict=True,
            **kwargs,
        )

        return output

    def generate(
        self, model, input_ids=None, **kwargs,
    ):

        bsz = input_ids.shape[0]

        past_prefix_prompts = self.get_prompt(bsz=bsz)

        inputs = {
            "input_ids": input_ids,
            "past_prefix_prompts": past_prefix_prompts,
        }
        output = model.generate(**inputs, **kwargs)

        return output

    def adversarial(
        self,
        model,
        input_ids=None,
        labels=None,
        labels_attention=None,
        past_key_values=None,
        discriminator=None,
        discriminator_label=None,
        feature_boundary_ids=None,
        discriminator_weight=None,
        whole_word_ids=None,
        attention_mask=None,
        train_discriminator=False,
        return_hidden_state=True,
        reduce_loss=False,
        **kwargs,
    ):
        bsz = input_ids.shape[0]

        past_prefix_prompts = self.get_prompt(bsz=bsz)

        past_prefix_prompts_encoder = [
            past_prefix_prompt[:3] for past_prefix_prompt in past_prefix_prompts
        ]

        encoder_outputs = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            whole_word_ids=whole_word_ids,
            past_prefix_prompts=past_prefix_prompts_encoder,
        )
        hidden_states = encoder_outputs[0]

        B = hidden_states.size(0)
        user_embeddings = None
        for b in range(B):
            user_embedding = (
                hidden_states[b][
                    feature_boundary_ids[b][0] : feature_boundary_ids[b][1]
                ]
                .mean(dim=0)
                .unsqueeze(0)
            )
            if b == 0:
                user_embeddings = user_embedding
            else:
                user_embeddings = torch.cat([user_embeddings, user_embedding], dim=0)
        # B * embedding_dim
        assert user_embeddings is not None

        discriminator_loss = discriminator(user_embeddings, discriminator_label)

        decoder_input_ids = model._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert (
                labels is None
            ), "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]

        if attention_mask is None:
            attention_mask = input_ids.ne(model.config.pad_token_id).to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
        encoder_attention_mask = attention_mask

        # Decode
        past_prefix_prompts_decoder = [
            past_prefix_prompt[3:] for past_prefix_prompt in past_prefix_prompts
        ]
        decoder_outputs = model.decoder(
            input_ids=decoder_input_ids,
            past_prefix_prompts=past_prefix_prompts_decoder,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = decoder_outputs[0]

        assert model.config.tie_word_embeddings is True

        if model.config.tie_word_embeddings:
            sequence_output = sequence_output * (model.model_dim ** -0.5)

        lm_logits = model.lm_head(sequence_output)

        rec_loss = None
        if labels is not None:
            if reduce_loss:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
            rec_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        lm_mask = labels_attention != 0
        lm_mask = lm_mask.float()
        B, L = labels.size()
        rec_loss = rec_loss.view(B, L) * lm_mask
        rec_loss = (rec_loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)).mean()

        if not train_discriminator:
            loss = rec_loss - discriminator_weight * discriminator_loss
            # loss = -discriminator_weight * discriminator_loss
        else:
            loss = rec_loss + discriminator_weight * discriminator_loss

        return P5AdversarialSeq2SeqLMOutput(
            loss=loss,
            rec_loss=rec_loss,
            discriminator_loss=discriminator_loss,
            feature_embeddings=user_embeddings,
        )


class AttnFFNPrefixTuningT5(T5PreTrainedModelwithPrefix):
    """Classification Head for  transformer encoders"""

    def __init__(
        self, config, optim_prefix=False, preseqlen=5,
    ):
        super().__init__(config)
        self.preseqlen = preseqlen
        self.config = config

        self.match_n_head = self.config.num_heads
        self.match_n_layer = self.config.num_layers
        self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.input_tokens = torch.arange(self.preseqlen).long().cuda()

        # need: self.match_n_layer * 2 * self.n_emb
        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
        self.wte_cross = nn.Embedding(self.preseqlen, self.n_embd)

        self.query_embedding = nn.Embedding(self.preseqlen * 3, self.n_embd)
        self.query = nn.Linear(self.n_embd, self.n_embd)
        self.encoder_key = nn.Sequential(
            nn.Linear(self.n_embd, 800), nn.Tanh(), nn.Linear(800, self.n_embd),
        )
        self.encoder_value = nn.Sequential(
            nn.Linear(self.n_embd, 800), nn.Tanh(), nn.Linear(800, self.n_embd),
        )
        self.cross_key = nn.Sequential(
            nn.Linear(self.n_embd, 800), nn.Tanh(), nn.Linear(800, self.n_embd),
        )
        self.cross_value = nn.Sequential(
            nn.Linear(self.n_embd, 800), nn.Tanh(), nn.Linear(800, self.n_embd),
        )
        self.decoder_key = nn.Sequential(
            nn.Linear(self.n_embd, 800), nn.Tanh(), nn.Linear(800, self.n_embd),
        )
        self.decoder_value = nn.Sequential(
            nn.Linear(self.n_embd, 800), nn.Tanh(), nn.Linear(800, self.n_embd),
        )

        self.combination_function = nn.MultiheadAttention(
            self.n_embd, self.match_n_head, batch_first=True
        )

        self.attention_map = nn.Linear(
            self.n_embd, self.match_n_layer * 2 * self.n_embd
        )

        nn.init.xavier_normal_(self.encoder_key[0].weight)
        nn.init.xavier_normal_(self.encoder_key[2].weight)
        nn.init.xavier_normal_(self.encoder_value[0].weight)
        nn.init.xavier_normal_(self.encoder_value[2].weight)

        nn.init.xavier_normal_(self.cross_key[0].weight)
        nn.init.xavier_normal_(self.cross_key[2].weight)
        nn.init.xavier_normal_(self.cross_value[0].weight)
        nn.init.xavier_normal_(self.cross_value[2].weight)

        nn.init.xavier_normal_(self.decoder_key[0].weight)
        nn.init.xavier_normal_(self.decoder_key[2].weight)
        nn.init.xavier_normal_(self.decoder_value[0].weight)
        nn.init.xavier_normal_(self.decoder_value[2].weight)

        self.get_prompt = self.get_prompt_p5

        self.dropout = nn.Dropout(0.1)

    def get_prompt_p5(self, bsz=1, sample_size=1):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1)

        ######## encoder prefix prompt ########
        encoder_embedding = self.wte(input_tokens)
        encoder_key = self.encoder_key(encoder_embedding)
        encoder_value = self.encoder_value(encoder_embedding)
        encoder_query_embedding = self.query_embedding(input_tokens)
        encoder_query = self.query(encoder_query_embedding)
        past_key_values = self.attention_map(
            self.combination_function(encoder_query, encoder_key, encoder_value)[0]
        )  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        ######## decoder prefix prompt ########
        decoder_embedding = self.wte_enc(input_tokens)
        decoder_key = self.decoder_key(decoder_embedding)
        decoder_value = self.decoder_value(decoder_embedding)
        decoder_query_embedding = self.query_embedding(input_tokens + self.preseqlen)
        decoder_query = self.query(decoder_query_embedding)
        past_key_values_enc = self.attention_map(
            self.combination_function(decoder_query, decoder_key, decoder_value)[0]
        )  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        ######## cross prefix prompt ########
        cross_embedding = self.wte_cross(input_tokens)
        cross_key = self.cross_key(cross_embedding)
        cross_value = self.cross_value(cross_embedding)
        cross_query_embedding = self.query_embedding(input_tokens + 2 * self.preseqlen)
        cross_query = self.query(cross_query_embedding)
        past_key_values_cross = self.attention_map(
            self.combination_function(cross_query, cross_key, cross_value)[0]
        )  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values_cross.shape
        past_key_values_cross = past_key_values_cross.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_cross = self.dropout(past_key_values_cross)
        past_key_values_cross = past_key_values_cross.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            key_val_enc = past_key_values_enc[i]
            key_val_cross = past_key_values_cross[i]
            # encoder
            # decoder
            # cross attention
            temp_dict = [
                key_val[0].contiguous().cuda(),
                key_val[1].contiguous().cuda(),
                torch.zeros(bsz, seqlen).to(key_val.device).bool(),
                key_val_enc[0].contiguous().cuda(),
                key_val_enc[1].contiguous().cuda(),
                torch.zeros(bsz, seqlen).to(key_val_enc.device).bool(),
                key_val_cross[0].contiguous().cuda(),
                key_val_cross[1].contiguous().cuda(),
                torch.zeros(bsz, seqlen).to(key_val_cross.device).bool(),
            ]

            result.append(temp_dict)

        return result

    def forward(
        self, model, input_ids=None, past_key_values=None, **kwargs,
    ):

        bsz = input_ids.shape[0]

        past_key_values_prompt = self.get_prompt(bsz=bsz)

        past_prefix_prompts = past_key_values_prompt

        output = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            past_prefix_prompts=past_prefix_prompts,
            **kwargs,
        )

        return output

    def encoder(
        self, model, input_ids=None, past_key_values=None, **kwargs,
    ):
        encoder = model.get_encoder()

        bsz = input_ids.shape[0]

        past_key_values_prompts = self.get_prompt(bsz=bsz)

        past_prefix_prompts_encoder = [
            past_key_values_prompt[:3]
            for past_key_values_prompt in past_key_values_prompts
        ]

        output = encoder(
            input_ids=input_ids,
            past_key_values=past_key_values,
            past_prefix_prompts=past_prefix_prompts_encoder,
            return_dict=True,
            **kwargs,
        )

        return output

    def generate(
        self, model, input_ids=None, **kwargs,
    ):

        bsz = input_ids.shape[0]

        past_prefix_prompts = self.get_prompt(bsz=bsz)

        inputs = {
            "input_ids": input_ids,
            "past_prefix_prompts": past_prefix_prompts,
        }
        output = model.generate(**inputs, **kwargs)

        return output

    def adversarial(
        self,
        model,
        input_ids=None,
        labels=None,
        labels_attention=None,
        past_key_values=None,
        discriminator=None,
        discriminator_label=None,
        feature_boundary_ids=None,
        discriminator_weight=None,
        whole_word_ids=None,
        attention_mask=None,
        train_discriminator=False,
        return_hidden_state=True,
        reduce_loss=False,
        **kwargs,
    ):
        bsz = input_ids.shape[0]

        past_prefix_prompts = self.get_prompt(bsz=bsz)

        past_prefix_prompts_encoder = [
            past_prefix_prompt[:3] for past_prefix_prompt in past_prefix_prompts
        ]

        encoder_outputs = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            whole_word_ids=whole_word_ids,
            past_prefix_prompts=past_prefix_prompts_encoder,
        )
        hidden_states = encoder_outputs[0]

        B = hidden_states.size(0)
        user_embeddings = None
        for b in range(B):
            user_embedding = (
                hidden_states[b][
                    feature_boundary_ids[b][0] : feature_boundary_ids[b][1]
                ]
                .mean(dim=0)
                .unsqueeze(0)
            )
            if b == 0:
                user_embeddings = user_embedding
            else:
                user_embeddings = torch.cat([user_embeddings, user_embedding], dim=0)
        # B * embedding_dim
        assert user_embeddings is not None

        discriminator_loss = discriminator(user_embeddings, discriminator_label)

        decoder_input_ids = model._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert (
                labels is None
            ), "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]

        if attention_mask is None:
            attention_mask = input_ids.ne(model.config.pad_token_id).to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
        encoder_attention_mask = attention_mask

        # Decode
        past_prefix_prompts_decoder = [
            past_prefix_prompt[3:] for past_prefix_prompt in past_prefix_prompts
        ]
        decoder_outputs = model.decoder(
            input_ids=decoder_input_ids,
            past_prefix_prompts=past_prefix_prompts_decoder,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = decoder_outputs[0]

        assert model.config.tie_word_embeddings is True

        if model.config.tie_word_embeddings:
            sequence_output = sequence_output * (model.model_dim ** -0.5)

        lm_logits = model.lm_head(sequence_output)

        rec_loss = None
        if labels is not None:
            if reduce_loss:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
            rec_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        lm_mask = labels_attention != 0
        lm_mask = lm_mask.float()
        B, L = labels.size()
        rec_loss = rec_loss.view(B, L) * lm_mask
        rec_loss = (rec_loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)).mean()

        if not train_discriminator:
            loss = rec_loss - discriminator_weight * discriminator_loss
            # loss = -discriminator_weight * discriminator_loss
        else:
            loss = rec_loss + discriminator_weight * discriminator_loss

        return P5AdversarialSeq2SeqLMOutput(
            loss=loss,
            rec_loss=rec_loss,
            discriminator_loss=discriminator_loss,
            feature_embeddings=user_embeddings,
        )


class P5AdversarialSeq2SeqLMOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    rec_loss: Optional[torch.FloatTensor] = None
    discriminator_loss: Optional[torch.FloatTensor] = None
    upward_discriminator_loss: Optional[torch.FloatTensor] = None
    gender_loss: Optional[torch.FloatTensor] = None
    age_loss: Optional[torch.FloatTensor] = None
    occupation_loss: Optional[torch.FloatTensor] = None
    marital_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    feature_embeddings: Optional[torch.FloatTensor] = None


if __name__ == "__main__":
    """
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    config = model.config
    prefixmodel = PrefixTuningBart(config)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    ARTICLE_TO_SUMMARIZE = (
        "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
        "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
        "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    )
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

    out = model(**inputs, return_dict=True, use_cache=True)
    out = out.past_key_values

    result = prefixmodel.get_prompt_p5()

    print(result[0]["encoder"]["prev_key"].size())


    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    set_seed()
    """
    model = T5ForConditionalGenerationwithPrefix.from_pretrained("t5-small").cuda()
    # from transformers import T5ForConditionalGeneration

    # orig_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    print("model loaded")
    config = model.config

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    ARTICLE_TO_SUMMARIZE = ["What do you recommend in French?", "What is your name?"]
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], padding="longest", return_tensors="pt",)[
        "input_ids"
    ].cuda()
    labels = tokenizer(
        "<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt"
    ).input_ids.cuda()
    print("inputs tokenized")

    set_seed()

    prefix_model_one = AttnPrefixTuningT5(config).cuda()
    prefix_model_two = AttnPrefixTuningT5(config).cuda()
    prefix_model_three = AttnPrefixTuningT5(config).cuda()

    prompts_one = prefix_model_one.get_prompt()
    prompts_two = prefix_model_two.get_prompt()
    prompts_three = prefix_model_three.get_prompt()

    # for l in range(len(prompts_one)):
    #    for i in range(len(prompts_one[l])):
    #        print(prompts_one[l][i].size())

    promt_models = [prefix_model_one, prefix_model_two, prefix_model_three]

    multi_prompt = CFunctionPrompt(promt_models, 5)
    prompts = multi_prompt.combine_promts(inputs, model_indices=[0, 1, 2])

    print("len(prompts)")
    print(len(prompts))

    print("prompt combined sizes")
    for p in prompts:
        print("len(p)")
        print(len(p))
        print(type(p[0]))
        # print(p[0].size())

    """
    out = prefix_model(
        model,
        input_ids=inputs,
        # max_length=20,
        # num_beams=10,
        whole_word_ids=torch.arange(inputs.size(-1)).unsqueeze(0).cuda(),
        labels=labels,
    )
    print(out.keys())
    # out = tokenizer.batch_decode(out)
    # print(out)
    set_seed()
    out = orig_model.generate(input_ids=inputs, num_beams=10, max_length=20)
    out = tokenizer.batch_decode(out)
    print(out)


    from transformers import T5Tokenizer, T5ForConditionalGeneration

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    ARTICLE_TO_SUMMARIZE = (
        "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
        "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
        "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    )
    input_ids = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=512, return_tensors="pt")[
        "input_ids"
    ]
    labels = tokenizer(
        "<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt"
    ).input_ids
    outputs = model(input_ids=input_ids, labels=labels)
    """
