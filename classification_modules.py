# (c) 2021 NCSOFT Corporation & Korea University. All rights reserved.
import logging
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import GPT2Model, GPT2PreTrainedModel, GPT2LMHeadModel
from transformers import BartModel, BartPretrainedModel, BartForConditionalGeneration
from torch.nn import Sigmoid, Softmax

logger = logging.getLogger(__name__)


class SelfAttention(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        self.linear = nn.Linear(emb_dim, 1)
        self.softmax = torch.nn.Softmax(dim=-2)

    def forward(self, input):
        logits = self.linear(input)
        weights = self.softmax(logits)
        return (input * weights).sum(dim=-2)


class BasicAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, queries, keys, values=None):
        """
        Compute attention.

        Args:
            queries: B x query_len x dim (queries)
            keys: B x key_len x dim (keys)
            values: B x value_len x dim (values)
        """
        # attend over keys with queries to get weights
        l1 = torch.bmm(queries, keys.transpose(1, 2))
        l2 = self.softmax(l1)
        # multiply weights on values
        if values is None:
            values = keys
        lhs_emb = torch.bmm(l2, values)
        return lhs_emb


class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = BasicAttention()
        self.attn2 = BasicAttention()

    def forward(self, queries, keys, values):
        """
        Compute cross-attention.

        Args:
            queries: B x query_len x dim (queries)
            keys: B x key_len x dim (keys)
            values: B x value_len x dim (values)
        """
        # attend over keys with queries -> enhanced queries
        enhanced_queries = self.attn1(queries=queries, keys=keys)

        # attend over value with enhanced queries
        output = self.attn1(queries=enhanced_queries, keys=values)

        return output


class ConcatCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = BasicAttention()

    def forward(self, queries, keys, values):
        """
        Compute cross-attention.

        Args:
            queries: B x query_len x dim (queries)
            keys: B x key_len x dim (keys)
            values: B x value_len x dim (values)
        """
        # concatenate key and values
        kv = torch.cat([keys, values], dim=1)
        # attend over keys with queries -> enhanced queries
        output = self.attn1(queries=queries, keys=kv)

        return output


class ColBERTSim(nn.Module):
    # def __init__(self):
    #     super().__init__()
    #     self.attn1 = BasicAttention()

    def forward(self, cand_rep, ctxt_rep, mask_cand=None, mask_ctxt=None):
        """
        Compute cross-attention.

        Args:
            cand_rep: B x query x dim (queries)
            ctxt_rep: B x key_len x dim (keys)
        """
        bsz = ctxt_rep.size(0)
        n_cand = cand_rep.size(1)
        n_ctxt = ctxt_rep.size(1)
        ctxt_len = ctxt_rep.size(2)
        cand_len = cand_rep.size(2)

        ctxt = ctxt_rep.unsqueeze(1).unsqueeze(3).transpose(4, 5)
        cand = cand_rep.unsqueeze(2).unsqueeze(4)
        # get token-level interaction score
        scores_padded = (cand @ ctxt).view(bsz, n_cand, n_ctxt, cand_len, ctxt_len)
        # mask out padding candidate tokens
        mask_cand = mask_cand.unsqueeze(2).expand(-1, -1, n_ctxt, -1)
        scores_padded[~mask_cand] = torch.tensor(-99999, dtype=scores_padded.dtype)
        # reduce token-level scores along the candidate tokens
        scores_padded = scores_padded.max(3).values
        # mask out padding context tokens
        # reduce token scores
        denom = torch.sum(mask_ctxt, -1, keepdim=True).transpose(1,2)
        scores = torch.sum(scores_padded * mask_ctxt.unsqueeze(1), dim=-1) / denom

        return scores


class ConcatSummary(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(emb_dim * 7, 1)  # hiddensize, numclasses

    def forward(self, output):
        dropout_pooled_output = self.dropout(output)
        logits = self.summary(dropout_pooled_output)
        return logits


class Summary(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(emb_dim, 1)  # hiddensize, numclasses

    def forward(self, output):
        dropout_pooled_output = self.dropout(output)
        logits = self.summary(dropout_pooled_output)
        return logits


class GPT2PK_ctxt(GPT2LMHeadModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.concat_summary = ConcatSummary(emb_dim=config.n_embd)
        self.summary = Summary(emb_dim=config.n_embd)
        self.attn1 = nn.Linear(config.n_embd, 5)
        self.attn2 = nn.Linear(5, config.n_embd)  # Selected knowledge 개수만
        self.max_position = config.n_positions
        # self.paragraph_sum = nn.Linear(config.n_embd, 1)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            input_eos=None,
            token_type_ids=None,
            only_dial_input_ids=None,
            only_dial_token_type_ids=None,
            persona_input_ids=None,
            knowledge_input_ids=None,
            persona_can_idx=None,
            persona_grounding=None,
            knowledge_can_idx=None,
            knowledge_grounding=None,
            tot_knowledge=None,
            tot_knowledge_token_ids=None,
            tot_knowledge_eos=None,
            training=None,
            mc_token_ids=None):

        persona = 50259
        knowledge = 50260
        padding = 50261
        bos = 50256
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        persona_tensor = torch.tensor([persona]).to(device)
        knowledge_tensor = torch.tensor([knowledge]).to(device)
        bos_tensor = torch.tensor([bos]).to(device)
        outputs = tuple()
        dynamic_lm_logits = None
        persona_logits = None
        knowledge_logits = None
        lm_labels = None

        if input_eos is not None:
            lm_hidden_states = self.transformer(input_ids=input_ids, token_type_ids=token_type_ids)['last_hidden_state']
            batch, seq_len, embdim = lm_hidden_states.size()
            lm_hidden_states_eos_list = []
            for i in range(batch):
                lm_hidden_states_batch = lm_hidden_states[i]
                lm_eos_batch = input_eos[i]
                lm_hidden_states_eos = torch.index_select(lm_hidden_states_batch, -2, lm_eos_batch)
                lm_hidden_states_eos_list.append(lm_hidden_states_eos)
            lm_eos_rep = torch.stack(lm_hidden_states_eos_list)
            # print("lm eos rep: ", lm_eos_rep.size()) #batch, 1, embdim

            tot_knowledge_hidden_states = \
                self.transformer(input_ids=tot_knowledge, token_type_ids=tot_knowledge_token_ids)['last_hidden_state']
            # print("tot knowledge: ", tot_knowledge_hidden_states.size()) #batch, 5(# paragraph), seqlen, embdim
            tot_knowledge_eos_list = []
            for i in range(batch):
                tot_knowledge_hidden_states_batch = tot_knowledge_hidden_states[i]
                tot_knowledge_eos_batch = tot_knowledge_eos[i]
                # print("tot_knowledge_hid batch: ", tot_knowledge_hidden_states_batch.size(), tot_knowledge_eos_batch.size()) #5, seqlen, embdim / 5
                tot_knowledge_eos_list_batch = []
                for j in range(5):
                    tot_knowledge_eos_token = torch.index_select(tot_knowledge_hidden_states_batch[j], -2,
                                                                 tot_knowledge_eos_batch[j])
                    tot_knowledge_eos_list_batch.append(tot_knowledge_eos_token.squeeze())
                tot_knowledge_eos_batch_rep = torch.stack(tot_knowledge_eos_list_batch)
                tot_knowledge_eos_list.append(tot_knowledge_eos_batch_rep)
            tot_knowledge_eos_final = torch.stack(tot_knowledge_eos_list)
            knowledge_inctxt_attn = self.attn1(tot_knowledge_eos_final)
            knowledge_inctxt_eos_rep = self.attn2(knowledge_inctxt_attn)
            inctxt_states = torch.cat((lm_eos_rep, knowledge_inctxt_eos_rep), dim=1).type_as(input_ids)

            sigmoid = Sigmoid()
            # persona candidates
            num_persona_can = 5
            if persona_input_ids is not None:
                persona_emb = self.transformer(input_ids=persona_input_ids)['last_hidden_state']
                if persona_can_idx is not None:
                    persona_list = []
                    for batch_i in range(batch):
                        inctxt_eos_batch = inctxt_states[batch_i]  # 6, 768
                        persona_emb_batch = persona_emb[batch_i]
                        persona_can_idx_batch = persona_can_idx[batch_i]
                        persona_batch_list = []
                        for i in range(num_persona_can):
                            persona_selected = torch.index_select(persona_emb_batch[i], 0, persona_can_idx_batch[i])
                            final_rep_persona = torch.cat(
                                [inctxt_eos_batch.type_as(lm_eos_rep), persona_selected.type_as(lm_eos_rep)],
                                dim=0)  # 7,768
                            persona_batch_list.append(final_rep_persona)
                        persona_batch_list = torch.stack(persona_batch_list)
                        persona_list.append(persona_batch_list)
                    persona_rep = torch.stack(persona_list).view(batch * num_persona_can, -1)
                    persona_logits = self.concat_summary(persona_rep).view(batch, -1)
                    outputs = (persona_logits,)

                    persona_pred_sigmoid = sigmoid(persona_logits)
                    persona_pred_sigmoid = (persona_pred_sigmoid > 0.5).float()
                    all_persona_pred = []
                    selected_persona_idx = list()
                    for batch_idx, persona_batch in enumerate(torch.eq(persona_pred_sigmoid, 1)):
                        batch_list_idx = list()
                        batch_list = list()
                        for i, can in enumerate(persona_batch):
                            if can == True:
                                batch_list_idx.append(can)
                                persona_selected_now = persona_input_ids[batch_idx][i]
                                mask_persona = torch.ne(persona_selected_now, padding)
                                persona_selected_now = torch.masked_select(persona_selected_now, mask_persona)
                                batch_list.append(persona_selected_now[:-2])
                        all_persona_pred.append(batch_list)
                        selected_persona_idx.append(batch_list_idx)

            # knowledge candidates
            num_knowledge_can = 10
            if knowledge_input_ids is not None:
                knowledge_emb = self.transformer(input_ids=knowledge_input_ids)['last_hidden_state']
                if knowledge_can_idx is not None:
                    knowledge_list = []
                    for batch_i in range(batch):
                        inctxt_eos_batch = inctxt_states[batch_i]
                        knowledge_emb_batch = knowledge_emb[batch_i]
                        knowledge_can_idx_batch = knowledge_can_idx[batch_i]
                        knowledge_batch_list = []
                        for i in range(num_knowledge_can):
                            knowledge_selected = torch.index_select(knowledge_emb_batch[i], 0,
                                                                    knowledge_can_idx_batch[i])
                            final_rep_knowledge = torch.cat(
                                [inctxt_eos_batch.type_as(lm_eos_rep), knowledge_selected.type_as(lm_eos_rep)], dim=0)
                            knowledge_batch_list.append(final_rep_knowledge)
                        knowledge_batch_list = torch.stack(knowledge_batch_list)
                        knowledge_list.append(knowledge_batch_list)
                    knowledge_rep = torch.stack(knowledge_list).view(batch * num_knowledge_can, -1)
                    knowledge_logits = self.concat_summary(knowledge_rep).view(batch, -1)
                    outputs = (knowledge_logits,) + outputs
                    softmax = Softmax(dim=-1)
                    knowledge_softmax = softmax(knowledge_logits)
                    _, k_index_1 = torch.topk(knowledge_softmax, k=1, dim=-1)
                    all_knowledge_pred = []
                    for batch_i in range(batch):
                        knowledge_pred_idx = k_index_1[batch_i]
                        knowledge_pred = knowledge_input_ids[batch_i][knowledge_pred_idx]
                        mask_knowledge = torch.ne(knowledge_pred, padding)
                        knowledge_pred = torch.masked_select(knowledge_pred, mask_knowledge)
                        knowledge_pred = knowledge_pred[1:-2]  # delete bos, knowledge_st, eos
                        if knowledge_pred.size()[0] > 150:
                            knowledge_pred = knowledge_pred[:150]
                        all_knowledge_pred.append(knowledge_pred)

            final_input_list = []
            final_input_tti_list = []
            final_lm_label_list = []
            for batch_i in range(batch):
                only_dial_input_ids_batch = only_dial_input_ids[batch_i]
                only_dial_token_type_ids_batch = only_dial_token_type_ids[batch_i]
                mask_only_dial_input_ids_batch = torch.ne(only_dial_input_ids_batch, padding)
                mask_only_dial_tti_batch = torch.ne(only_dial_token_type_ids_batch, padding)
                only_dial_input_ids_batch = torch.masked_select(only_dial_input_ids_batch,
                                                                mask_only_dial_input_ids_batch)
                only_dial_token_type_ids_batch = torch.masked_select(only_dial_token_type_ids_batch,
                                                                     mask_only_dial_tti_batch)
                if len(all_persona_pred[batch_i]) > 0:
                    concat_persona = torch.cat(all_persona_pred[batch_i], dim=-1)
                    new_persona = torch.cat([persona_tensor, concat_persona], dim=-1)
                    new_persona_tti = torch.tensor([persona] * (new_persona.size()[0])).to(device)
                else:
                    new_persona = None
                    new_persona_tti = None

                new_knowledge = torch.cat([knowledge_tensor, all_knowledge_pred[batch_i]], dim=-1)
                new_knowledge_tti = torch.tensor([knowledge] * (new_knowledge.size()[0])).to(device)

                if new_persona is not None:
                    new_input = torch.cat([bos_tensor, new_knowledge, new_persona, only_dial_input_ids_batch], dim=-1)
                    new_input_tti = torch.cat(
                        [knowledge_tensor, new_knowledge_tti, new_persona_tti, only_dial_token_type_ids_batch], dim=-1)
                else:
                    new_input = torch.cat([bos_tensor, new_knowledge, only_dial_input_ids_batch], dim=-1)
                    new_input_tti = torch.cat([knowledge_tensor, new_knowledge_tti, only_dial_token_type_ids_batch],
                                              dim=-1)

                new_input_size = new_input.size()[0]
                new_lm_label = torch.cat(
                    [torch.tensor([-100] * (new_input_size - (only_dial_input_ids_batch.size()[0]) + 1)).to(device),
                     only_dial_input_ids_batch[1:]], dim=-1)
                assert new_input.size() == new_input_tti.size() == new_lm_label.size()
                if new_input_size < int(self.max_position):
                    padding_size = int(self.max_position) - new_input_size
                    add_padding = torch.tensor([padding] * padding_size).to(device)
                    add_lm_padding = torch.tensor([-100] * padding_size).to(device)
                    final_input = torch.cat([new_input, add_padding], dim=-1)
                    final_tti_input = torch.cat([new_input_tti, add_padding], dim=-1)
                    final_lm_label = torch.cat([new_lm_label, add_lm_padding], dim=-1)
                final_input_list.append(final_input)
                final_input_tti_list.append(final_tti_input)
                final_lm_label_list.append(final_lm_label)
            input_ids = torch.stack(final_input_list)
            token_type_ids = torch.stack(final_input_tti_list)
            lm_labels = torch.stack(final_lm_label_list)

        dynamic_lm_hidden_states = self.transformer(input_ids=input_ids, token_type_ids=token_type_ids)[
            'last_hidden_state']
        if dynamic_lm_hidden_states is not None:
            dynamic_lm_logits = self.lm_head(dynamic_lm_hidden_states)
            outputs = (dynamic_lm_logits,) + outputs

        if persona_grounding is not None:
            loss_fct = BCEWithLogitsLoss()
            persona_loss = loss_fct(persona_logits.view(batch, -1), persona_grounding.type_as(persona_logits))
            outputs = (persona_loss,) + outputs

        if knowledge_grounding is not None:
            loss_fct = CrossEntropyLoss()
            knowledge_loss = loss_fct(knowledge_logits.view(batch, -1), knowledge_grounding)
            outputs = (knowledge_loss,) + outputs

        if training is not True:
            outputs = (lm_labels,) + outputs
            lm_labels = None

        if lm_labels is not None:
            shift_logits = dynamic_lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (lm_loss,) + outputs

        return outputs  # (lm_loss-training), (lm_label-validation), (knowledge_loss), (persona_loss), dynamic_lm_logits, knowledge_logits, persona_logits, presents, (all hidden_states), (attentions)


class BARTPK_ctxt(BartForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        # config.vocab_size = config.vocab_size + 4
        self.model = BartModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.concat_summary = ConcatSummary(emb_dim=config.d_model)
        self.summary = Summary(emb_dim=config.d_model)
        self.attn1 = nn.Linear(config.d_model, 5)
        self.attn2 = nn.Linear(5, config.d_model)  # Selected knowledge 개수만
        self.max_position = config.max_position_embeddings
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            input_eos=None,
            only_dial_input_ids=None,
            decoder_input_ids=None,
            persona_input_ids=None,
            knowledge_input_ids=None,
            persona_can_idx=None,
            persona_grounding=None,
            knowledge_can_idx=None,
            knowledge_grounding=None,
            tot_knowledge=None,
            tot_knowledge_eos=None,
            training=None,
            lm_labels=None,
            mc_token_ids=None):

        # machine = 50265
        # human = 50266
        persona = 50267
        knowledge = 50268
        padding = 1
        bos = 0
        eos = 2
        num_chosen_paragraph = 5
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        persona_tensor = torch.tensor([persona]).to(device)
        knowledge_tensor = torch.tensor([knowledge]).to(device)
        bos_tensor = torch.tensor([bos]).to(device)
        eos_tensor = torch.tensor([eos]).to(device)

        outputs = tuple()
        dynamic_lm_logits = None
        persona_logits = None
        knowledge_logits = None

        if input_eos is not None:
            lm_hidden_states = self.model(input_ids=input_ids)['last_hidden_state']
            batch, seq_len, embdim = lm_hidden_states.size()
            lm_hidden_states_eos_list = []
            for i in range(batch):
                lm_hidden_states_batch = lm_hidden_states[i]
                lm_eos_batch = input_eos[i]
                lm_hidden_states_eos = torch.index_select(lm_hidden_states_batch, -2, lm_eos_batch)
                lm_hidden_states_eos_list.append(lm_hidden_states_eos)
            lm_eos_rep = torch.stack(lm_hidden_states_eos_list)

            tot_knowledge_hidden_states = self.model(input_ids=tot_knowledge.view(batch * num_chosen_paragraph, -1))[
                'last_hidden_state'].view(batch, num_chosen_paragraph, -1, embdim)
            tot_knowledge_eos_list = []
            for i in range(batch):
                tot_knowledge_hidden_states_batch = tot_knowledge_hidden_states[i]
                tot_knowledge_eos_batch = tot_knowledge_eos[i]
                tot_knowledge_eos_list_batch = []
                for j in range(5):
                    tot_knowledge_eos_token = torch.index_select(tot_knowledge_hidden_states_batch[j], -2,
                                                                 tot_knowledge_eos_batch[j])
                    tot_knowledge_eos_list_batch.append(tot_knowledge_eos_token.squeeze())
                tot_knowledge_eos_batch_rep = torch.stack(tot_knowledge_eos_list_batch)
                tot_knowledge_eos_list.append(tot_knowledge_eos_batch_rep)
            tot_knowledge_eos_final = torch.stack(tot_knowledge_eos_list)
            knowledge_inctxt_attn = self.attn1(tot_knowledge_eos_final)
            knowledge_inctxt_eos_rep = self.attn2(knowledge_inctxt_attn)
            inctxt_states = torch.cat((lm_eos_rep, knowledge_inctxt_eos_rep), dim=1).type_as(input_ids)

            sigmoid = Sigmoid()

            # persona candidates
            num_persona_can = 5
            if persona_input_ids is not None:
                persona_emb = self.model(input_ids=persona_input_ids.view(batch * num_persona_can, -1))[
                    'last_hidden_state'].view(batch, num_persona_can, -1, embdim)
                if persona_can_idx is not None:
                    persona_list = []
                    for batch_i in range(batch):
                        inctxt_eos_batch = inctxt_states[batch_i]
                        persona_emb_batch = persona_emb[batch_i]
                        persona_can_idx_batch = persona_can_idx[batch_i]
                        persona_batch_list = []
                        for i in range(num_persona_can):
                            persona_selected = torch.index_select(persona_emb_batch[i], 0, persona_can_idx_batch[i])
                            final_rep_persona = torch.cat(
                                [inctxt_eos_batch.type_as(lm_eos_rep), persona_selected.type_as(lm_eos_rep)], dim=0)
                            persona_batch_list.append(final_rep_persona)
                        persona_batch_list = torch.stack(persona_batch_list)
                        persona_list.append(persona_batch_list)
                    persona_rep = torch.stack(persona_list).view(batch * num_persona_can, -1)
                    persona_logits = self.concat_summary(persona_rep).view(batch, -1)
                    outputs = (persona_logits,)

                    persona_pred_sigmoid = sigmoid(persona_logits)
                    persona_pred_sigmoid = (persona_pred_sigmoid > 0.5).float()
                    all_persona_pred = []
                    selected_persona_idx = list()
                    for batch_idx, persona_batch in enumerate(torch.eq(persona_pred_sigmoid, 1)):
                        batch_list_idx = list()
                        batch_list = list()
                        for i, can in enumerate(persona_batch):
                            if can == True:
                                batch_list_idx.append(can)
                                persona_selected_now = persona_input_ids[batch_idx][i]
                                mask_persona = torch.ne(persona_selected_now, padding)
                                persona_selected_now = torch.masked_select(persona_selected_now, mask_persona)
                                batch_list.append(persona_selected_now[:-2])
                        all_persona_pred.append(batch_list)
                        selected_persona_idx.append(batch_list_idx)

            # knowledge candidates
            num_knowledge_can = 10
            if knowledge_input_ids is not None:
                knowledge_emb = self.model(input_ids=knowledge_input_ids.view(batch * num_knowledge_can, -1))[
                    'last_hidden_state'].view(batch, num_knowledge_can, -1, embdim)
                if knowledge_can_idx is not None:
                    knowledge_list = []
                    for batch_i in range(batch):
                        inctxt_eos_batch = inctxt_states[batch_i]
                        knowledge_emb_batch = knowledge_emb[batch_i]
                        knowledge_can_idx_batch = knowledge_can_idx[batch_i]
                        knowledge_batch_list = []
                        for i in range(num_knowledge_can):
                            knowledge_selected = torch.index_select(knowledge_emb_batch[i], 0,
                                                                    knowledge_can_idx_batch[i])
                            final_rep_knowledge = torch.cat(
                                [inctxt_eos_batch.type_as(lm_eos_rep), knowledge_selected.type_as(lm_eos_rep)], dim=0)
                            knowledge_batch_list.append(final_rep_knowledge)
                        knowledge_batch_list = torch.stack(knowledge_batch_list)
                        knowledge_list.append(knowledge_batch_list)
                    knowledge_rep = torch.stack(knowledge_list).view(batch * num_knowledge_can, -1)
                    knowledge_logits = self.concat_summary(knowledge_rep).view(batch, -1)
                    outputs = (knowledge_logits,) + outputs
                    softmax = Softmax(dim=-1)
                    knowledge_softmax = softmax(knowledge_logits)
                    _, k_index_1 = torch.topk(knowledge_softmax, k=1, dim=-1)
                    all_knowledge_pred = []
                    for batch_i in range(batch):
                        knowledge_pred_idx = k_index_1[batch_i]
                        knowledge_pred = knowledge_input_ids[batch_i][knowledge_pred_idx]
                        mask_knowledge = torch.ne(knowledge_pred, padding)
                        knowledge_pred = torch.masked_select(knowledge_pred, mask_knowledge)
                        knowledge_pred = knowledge_pred[1:-2]
                        all_knowledge_pred.append(knowledge_pred)  # delete bos, knowledge_st, eos

            final_input_list = []
            for batch_i in range(batch):
                only_dial_input_ids_batch = only_dial_input_ids[batch_i]
                mask_only_dial_input_ids_batch = torch.ne(only_dial_input_ids_batch, padding)
                only_dial_input_ids_batch = torch.masked_select(only_dial_input_ids_batch,
                                                                mask_only_dial_input_ids_batch)
                if len(all_persona_pred[batch_i]) > 0:
                    concat_persona = torch.cat(all_persona_pred[batch_i], dim=-1)
                    new_persona = torch.cat([persona_tensor, concat_persona], dim=-1)
                else:
                    new_persona = None

                new_knowledge = torch.cat([knowledge_tensor, all_knowledge_pred[batch_i]], dim=-1)

                if new_persona is not None:
                    new_input = torch.cat(
                        [bos_tensor, new_knowledge, new_persona, only_dial_input_ids_batch, eos_tensor], dim=-1)
                else:
                    new_input = torch.cat([bos_tensor, new_knowledge, only_dial_input_ids_batch, eos_tensor], dim=-1)

                new_input_size = new_input.size()[0]

                if new_input_size < int(self.max_position):
                    padding_size = int(self.max_position) - new_input_size
                    add_padding = torch.tensor([padding] * padding_size).to(device)
                    final_input = torch.cat([new_input, add_padding], dim=-1)
                final_input_list.append(final_input)
            input_ids = torch.stack(final_input_list)
        dynamic_lm_hidden_states = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)[
            'last_hidden_state']

        if dynamic_lm_hidden_states is not None:
            dynamic_lm_logits = self.lm_head(dynamic_lm_hidden_states)
            outputs = (dynamic_lm_logits,) + outputs

        if persona_grounding is not None:
            loss_fct = BCEWithLogitsLoss()
            persona_loss = loss_fct(persona_logits.view(batch, -1), persona_grounding.type_as(persona_logits))
            outputs = (persona_loss,) + outputs

        if knowledge_grounding is not None:
            loss_fct = CrossEntropyLoss()
            knowledge_loss = loss_fct(knowledge_logits.view(batch, -1), knowledge_grounding)
            outputs = (knowledge_loss,) + outputs

        if training is True:
            shift_logits = dynamic_lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (lm_loss,) + outputs

        return outputs  # (lm_loss-training), (knowledge_loss), (persona_loss), dynamic_lm_logits, knowledge_logits, persona_logits, persona_detect_logits, presents, (all hidden_states), (attentions)


class GPT2PK_cratn(GPT2LMHeadModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.concat_summary = ConcatSummary(emb_dim=config.n_embd)

        self.cross_attn_persona = CrossAttention()
        self.cross_attn_knowledge = CrossAttention()
        self.summary_persona = Summary(emb_dim=config.n_embd)
        self.summary_knowledge = Summary(emb_dim=config.n_embd)
        self.sigmoid = Sigmoid()
        self.softmax = Softmax(dim=-1)
        self.max_position = config.n_positions
        # self.paragraph_sum = nn.Linear(config.n_embd, 1)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            input_eos=None,
            token_type_ids=None,
            only_dial_input_ids=None,
            only_dial_token_type_ids=None,
            persona_input_ids=None,
            knowledge_input_ids=None,
            persona_can_idx=None,
            persona_grounding=None,
            knowledge_can_idx=None,
            knowledge_grounding=None,
            tot_knowledge=None,
            tot_knowledge_token_ids=None,
            tot_knowledge_eos=None,
            training=None,
            mc_token_ids=None):

        persona = 50259
        knowledge = 50260
        padding = 50261
        bos = 50256
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        persona_tensor = torch.tensor([persona]).to(device)
        knowledge_tensor = torch.tensor([knowledge]).to(device)
        bos_tensor = torch.tensor([bos]).to(device)
        outputs = tuple()
        dynamic_lm_logits = None
        persona_logits = None
        knowledge_logits = None
        lm_labels = None

        if input_eos is not None:
            ######################################################################################
            # encode with transformers
            utterance_emb = self.transformer(input_ids=only_dial_input_ids)['last_hidden_state']
            persona_emb = self.transformer(input_ids=persona_input_ids)['last_hidden_state']
            knowledge_emb = self.transformer(input_ids=knowledge_input_ids)['last_hidden_state']

            # get the final persona sentence embedding
            # persona_emb_r = self.self_attn(persona_emb)
            batch, n_persona, seq_leng, n_embd = persona_emb.size()
            i = torch.arange(batch * n_persona).reshape(batch * n_persona, 1, 1)
            j = persona_can_idx.reshape(batch * n_persona, 1, 1)
            k = torch.arange(n_embd)
            persona_emb_final = persona_emb.view(batch * n_persona, seq_leng, n_embd)[i, j, k]
            persona_emb_final = persona_emb_final.reshape(batch, n_persona, n_embd)

            # get the final persona sentence embedding
            batch, n_knowledge, seq_leng, n_embd = knowledge_emb.size()
            i = torch.arange(batch * n_knowledge).reshape(batch * n_knowledge, 1, 1)
            j = knowledge_can_idx.reshape(batch * n_knowledge, 1, 1)
            k = torch.arange(n_embd)
            knowledge_emb_final = knowledge_emb.view(batch * n_knowledge, seq_leng, n_embd)[i, j, k]
            knowledge_emb_final = knowledge_emb_final.reshape(batch, n_knowledge, n_embd)

            # reduce persona/knowledge embeddings
            persona_emb_crossed = self.cross_attn_persona(queries=persona_emb_final, keys=knowledge_emb_final,
                                                          values=utterance_emb)
            knowledge_emb_crossed = self.cross_attn_knowledge(queries=knowledge_emb_final, keys=persona_emb_final,
                                                              values=utterance_emb)

            # pick persona
            persona_logits = self.summary_persona(persona_emb_crossed + persona_emb_final).view(batch, -1)
            outputs = (persona_logits,) + outputs
            persona_pred_sigmoid = self.sigmoid(persona_logits)
            persona_pred_sigmoid = (persona_pred_sigmoid > 0.5).float()
            all_persona_pred = []
            selected_persona_idx = list()
            for batch_idx, persona_batch in enumerate(torch.eq(persona_pred_sigmoid, 1)):
                batch_list_idx = list()
                batch_list = list()
                for i, can in enumerate(persona_batch):
                    if can == True:
                        batch_list_idx.append(can)
                        persona_selected_now = persona_input_ids[batch_idx][i]
                        mask_persona = torch.ne(persona_selected_now, padding)
                        persona_selected_now = torch.masked_select(persona_selected_now, mask_persona)
                        batch_list.append(persona_selected_now[:-2])
                all_persona_pred.append(batch_list)
                selected_persona_idx.append(batch_list_idx)

            # pick knowledge
            knowledge_logits = self.summary_knowledge(knowledge_emb_crossed + knowledge_emb_final).view(batch, -1)
            outputs = (knowledge_logits,) + outputs
            knowledge_softmax = self.softmax(knowledge_logits)
            _, k_index_1 = torch.topk(knowledge_softmax, k=1, dim=-1)
            all_knowledge_pred = []
            for batch_i in range(batch):
                knowledge_pred_idx = k_index_1[batch_i]
                knowledge_pred = knowledge_input_ids[batch_i][knowledge_pred_idx]
                mask_knowledge = torch.ne(knowledge_pred, padding)
                knowledge_pred = torch.masked_select(knowledge_pred, mask_knowledge)
                knowledge_pred = knowledge_pred[1:-2]  # delete bos, knowledge_st, eos
                if knowledge_pred.size()[0] > 150:
                    knowledge_pred = knowledge_pred[:150]
                all_knowledge_pred.append(knowledge_pred)

            final_input_list = []
            final_input_tti_list = []
            final_lm_label_list = []
            for batch_i in range(batch):
                only_dial_input_ids_batch = only_dial_input_ids[batch_i]
                only_dial_token_type_ids_batch = only_dial_token_type_ids[batch_i]
                mask_only_dial_input_ids_batch = torch.ne(only_dial_input_ids_batch, padding)
                mask_only_dial_tti_batch = torch.ne(only_dial_token_type_ids_batch, padding)
                only_dial_input_ids_batch = torch.masked_select(only_dial_input_ids_batch,
                                                                mask_only_dial_input_ids_batch)
                only_dial_token_type_ids_batch = torch.masked_select(only_dial_token_type_ids_batch,
                                                                     mask_only_dial_tti_batch)
                if len(all_persona_pred[batch_i]) > 0:
                    concat_persona = torch.cat(all_persona_pred[batch_i], dim=-1)
                    new_persona = torch.cat([persona_tensor, concat_persona], dim=-1)
                    new_persona_tti = torch.tensor([persona] * (new_persona.size()[0])).to(device)
                else:
                    new_persona = None
                    new_persona_tti = None

                new_knowledge = torch.cat([knowledge_tensor, all_knowledge_pred[batch_i]], dim=-1)
                new_knowledge_tti = torch.tensor([knowledge] * (new_knowledge.size()[0])).to(device)

                if new_persona is not None:
                    new_input = torch.cat([bos_tensor, new_knowledge, new_persona, only_dial_input_ids_batch], dim=-1)
                    new_input_tti = torch.cat(
                        [knowledge_tensor, new_knowledge_tti, new_persona_tti, only_dial_token_type_ids_batch], dim=-1)
                else:
                    new_input = torch.cat([bos_tensor, new_knowledge, only_dial_input_ids_batch], dim=-1)
                    new_input_tti = torch.cat([knowledge_tensor, new_knowledge_tti, only_dial_token_type_ids_batch],
                                              dim=-1)

                new_input_size = new_input.size()[0]
                new_lm_label = torch.cat(
                    [torch.tensor([-100] * (new_input_size - (only_dial_input_ids_batch.size()[0]) + 1)).to(device),
                     only_dial_input_ids_batch[1:]], dim=-1)

                assert new_input.size() == new_input_tti.size() == new_lm_label.size()
                if new_input_size < int(self.max_position):
                    padding_size = int(self.max_position) - new_input_size
                    add_padding = torch.tensor([padding] * padding_size).to(device)
                    add_lm_padding = torch.tensor([-100] * padding_size).to(device)
                    final_input = torch.cat([new_input, add_padding], dim=-1)
                    final_tti_input = torch.cat([new_input_tti, add_padding], dim=-1)
                    final_lm_label = torch.cat([new_lm_label, add_lm_padding], dim=-1)
                final_input_list.append(final_input)
                final_input_tti_list.append(final_tti_input)
                final_lm_label_list.append(final_lm_label)
            input_ids = torch.stack(final_input_list)
            token_type_ids = torch.stack(final_input_tti_list)
            lm_labels = torch.stack(final_lm_label_list)

        dynamic_lm_hidden_states = self.transformer(input_ids=input_ids, token_type_ids=token_type_ids)[
            'last_hidden_state']
        if dynamic_lm_hidden_states is not None:
            dynamic_lm_logits = self.lm_head(dynamic_lm_hidden_states)
            outputs = (dynamic_lm_logits,) + outputs

        if persona_grounding is not None:
            loss_fct = BCEWithLogitsLoss()
            persona_loss = loss_fct(persona_logits.view(batch, -1), persona_grounding.type_as(persona_logits))
            outputs = (persona_loss,) + outputs

        if knowledge_grounding is not None:
            loss_fct = CrossEntropyLoss()
            knowledge_loss = loss_fct(knowledge_logits.view(batch, -1), knowledge_grounding)
            outputs = (knowledge_loss,) + outputs

        if training is not True:
            outputs = (lm_labels,) + outputs
            lm_labels = None

        if lm_labels is not None:
            shift_logits = dynamic_lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (lm_loss,) + outputs

        # train: lm_loss, knowledge_loss, persona_loss, dynamic_lm_logits, knowledge_logits, persona_logits
        # valid: lm_label, dynamic_lm_logits, knowledge_logits, persona_logits
        return outputs


class BARTPK_cratn(BartForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        # config.vocab_size = config.vocab_size + 4
        self.model = BartModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.concat_summary = ConcatSummary(emb_dim=config.d_model)
        self.summary = Summary(emb_dim=config.d_model)
        self.attn1 = nn.Linear(config.d_model, 5)
        self.attn2 = nn.Linear(5, config.d_model)  # Selected knowledge 개수만
        self.cross_attn_persona = CrossAttention()
        self.cross_attn_knowledge = CrossAttention()
        self.summary_persona = Summary(emb_dim=config.d_model)
        self.summary_knowledge = Summary(emb_dim=config.d_model)
        self.sigmoid = Sigmoid()
        self.softmax = Softmax(dim=-1)
        self.max_position = config.max_position_embeddings
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            input_eos=None,
            only_dial_input_ids=None,
            decoder_input_ids=None,
            persona_input_ids=None,
            knowledge_input_ids=None,
            persona_can_idx=None,
            persona_grounding=None,
            knowledge_can_idx=None,
            knowledge_grounding=None,
            tot_knowledge=None,
            tot_knowledge_eos=None,
            training=None,
            lm_labels=None,
            mc_token_ids=None):

        # machine = 50265
        # human = 50266
        persona = 50267
        knowledge = 50268
        padding = 1
        bos = 0
        eos = 2
        num_chosen_paragraph = 5
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        persona_tensor = torch.tensor([persona]).to(device)
        knowledge_tensor = torch.tensor([knowledge]).to(device)
        bos_tensor = torch.tensor([bos]).to(device)
        eos_tensor = torch.tensor([eos]).to(device)

        outputs = tuple()
        dynamic_lm_logits = None
        persona_logits = None
        knowledge_logits = None

        if input_eos is not None:
            ######################################################################################
            # encode with BART
            utterance_emb = self.model(input_ids=only_dial_input_ids)['last_hidden_state']
            batch, seq_len, embdim = utterance_emb.size()
            num_persona_can = 5
            persona_emb = self.model(input_ids=persona_input_ids.view(batch * num_persona_can, -1))[
                'last_hidden_state'].view(batch, num_persona_can, -1, embdim)
            num_knowledge_can = 10
            knowledge_emb = self.model(input_ids=knowledge_input_ids.view(batch * num_knowledge_can, -1))[
                'last_hidden_state'].view(batch, num_knowledge_can, -1, embdim)

            # get the final persona sentence embedding
            # persona_emb_r = self.self_attn(persona_emb)
            batch, n_persona, seq_leng, n_embd = persona_emb.size()
            i = torch.arange(batch * n_persona).reshape(batch * n_persona, 1, 1)
            j = persona_can_idx.reshape(batch * n_persona, 1, 1)
            k = torch.arange(n_embd)
            persona_emb_final = persona_emb.view(batch * n_persona, seq_leng, n_embd)[i, j, k]
            persona_emb_final = persona_emb_final.reshape(batch, n_persona, n_embd)

            # get the final persona sentence embedding
            batch, n_knowledge, seq_leng, n_embd = knowledge_emb.size()
            i = torch.arange(batch * n_knowledge).reshape(batch * n_knowledge, 1, 1)
            j = knowledge_can_idx.reshape(batch * n_knowledge, 1, 1)
            k = torch.arange(n_embd)
            knowledge_emb_final = knowledge_emb.view(batch * n_knowledge, seq_leng, n_embd)[i, j, k]
            knowledge_emb_final = knowledge_emb_final.reshape(batch, n_knowledge, n_embd)

            # reduce persona/knowledge embeddings
            persona_emb_crossed = self.cross_attn_persona(queries=persona_emb_final, keys=knowledge_emb_final,
                                                          values=utterance_emb)
            knowledge_emb_crossed = self.cross_attn_knowledge(queries=knowledge_emb_final, keys=persona_emb_final,
                                                              values=utterance_emb)

            # pick persona
            persona_logits = self.summary_persona(persona_emb_crossed + persona_emb_final).view(batch, -1)
            outputs = (persona_logits,) + outputs

            persona_pred_sigmoid = self.sigmoid(persona_logits)
            persona_pred_sigmoid = (persona_pred_sigmoid > 0.5).float()
            all_persona_pred = []
            selected_persona_idx = list()
            for batch_idx, persona_batch in enumerate(torch.eq(persona_pred_sigmoid, 1)):
                batch_list_idx = list()
                batch_list = list()
                for i, can in enumerate(persona_batch):
                    if can == True:
                        batch_list_idx.append(can)
                        persona_selected_now = persona_input_ids[batch_idx][i]
                        mask_persona = torch.ne(persona_selected_now, padding)
                        persona_selected_now = torch.masked_select(persona_selected_now, mask_persona)
                        batch_list.append(persona_selected_now[:-2])
                all_persona_pred.append(batch_list)
                selected_persona_idx.append(batch_list_idx)

            # pick knowledge
            knowledge_logits = self.summary_knowledge(knowledge_emb_crossed + knowledge_emb_final).view(batch, -1)
            outputs = (knowledge_logits,) + outputs
            knowledge_softmax = self.softmax(knowledge_logits)
            _, k_index_1 = torch.topk(knowledge_softmax, k=1, dim=-1)
            all_knowledge_pred = []
            for batch_i in range(batch):
                knowledge_pred_idx = k_index_1[batch_i]
                knowledge_pred = knowledge_input_ids[batch_i][knowledge_pred_idx]
                mask_knowledge = torch.ne(knowledge_pred, padding)
                knowledge_pred = torch.masked_select(knowledge_pred, mask_knowledge)
                knowledge_pred = knowledge_pred[1:-2]
                all_knowledge_pred.append(knowledge_pred)  # delete bos, knowledge_st, eos

            # construct final input
            final_input_list = []
            for batch_i in range(batch):
                only_dial_input_ids_batch = only_dial_input_ids[batch_i]
                mask_only_dial_input_ids_batch = torch.ne(only_dial_input_ids_batch, padding)
                only_dial_input_ids_batch = torch.masked_select(only_dial_input_ids_batch,
                                                                mask_only_dial_input_ids_batch)
                if len(all_persona_pred[batch_i]) > 0:
                    concat_persona = torch.cat(all_persona_pred[batch_i], dim=-1)
                    new_persona = torch.cat([persona_tensor, concat_persona], dim=-1)
                else:
                    new_persona = None

                new_knowledge = torch.cat([knowledge_tensor, all_knowledge_pred[batch_i]], dim=-1)

                if new_persona is not None:
                    new_input = torch.cat(
                        [bos_tensor, new_knowledge, new_persona, only_dial_input_ids_batch, eos_tensor], dim=-1)
                else:
                    new_input = torch.cat([bos_tensor, new_knowledge, only_dial_input_ids_batch, eos_tensor], dim=-1)

                new_input_size = new_input.size()[0]

                if new_input_size < int(self.max_position):
                    padding_size = int(self.max_position) - new_input_size
                    add_padding = torch.tensor([padding] * padding_size).to(device)
                    final_input = torch.cat([new_input, add_padding], dim=-1)
                final_input_list.append(final_input)
            input_ids = torch.stack(final_input_list)

        # call BART model
        dynamic_lm_hidden_states = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)[
            'last_hidden_state']

        if dynamic_lm_hidden_states is not None:
            dynamic_lm_logits = self.lm_head(dynamic_lm_hidden_states)
            outputs = (dynamic_lm_logits,) + outputs

        if persona_grounding is not None:
            loss_fct = BCEWithLogitsLoss()
            persona_loss = loss_fct(persona_logits.view(batch, -1), persona_grounding.type_as(persona_logits))
            outputs = (persona_loss,) + outputs

        if knowledge_grounding is not None:
            loss_fct = CrossEntropyLoss()
            knowledge_loss = loss_fct(knowledge_logits.view(batch, -1), knowledge_grounding)
            outputs = (knowledge_loss,) + outputs

        if training is True:
            shift_logits = dynamic_lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (lm_loss,) + outputs

        return outputs  # (lm_loss-training), (knowledge_loss), (persona_loss), dynamic_lm_logits, knowledge_logits, persona_logits, persona_detect_logits, presents, (all hidden_states), (attentions)


class GPT2PK_catatn(GPT2LMHeadModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.concat_summary = ConcatSummary(emb_dim=config.n_embd)

        self.cross_attn_persona = ConcatCrossAttention()
        self.cross_attn_knowledge = ConcatCrossAttention()
        self.summary_persona = Summary(emb_dim=config.n_embd)
        self.summary_knowledge = Summary(emb_dim=config.n_embd)
        self.sigmoid = Sigmoid()
        self.softmax = Softmax(dim=-1)
        self.max_position = config.n_positions
        # self.paragraph_sum = nn.Linear(config.n_embd, 1)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            input_eos=None,
            token_type_ids=None,
            only_dial_input_ids=None,
            only_dial_token_type_ids=None,
            persona_input_ids=None,
            knowledge_input_ids=None,
            persona_can_idx=None,
            persona_grounding=None,
            knowledge_can_idx=None,
            knowledge_grounding=None,
            tot_knowledge=None,
            tot_knowledge_token_ids=None,
            tot_knowledge_eos=None,
            training=None,
            mc_token_ids=None):

        persona = 50259
        knowledge = 50260
        padding = 50261
        bos = 50256
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        persona_tensor = torch.tensor([persona]).to(device)
        knowledge_tensor = torch.tensor([knowledge]).to(device)
        bos_tensor = torch.tensor([bos]).to(device)
        outputs = tuple()
        dynamic_lm_logits = None
        persona_logits = None
        knowledge_logits = None
        lm_labels = None

        if input_eos is not None:
            ######################################################################################
            # encode with transformers
            utterance_emb = self.transformer(input_ids=only_dial_input_ids)['last_hidden_state']
            persona_emb = self.transformer(input_ids=persona_input_ids)['last_hidden_state']
            knowledge_emb = self.transformer(input_ids=knowledge_input_ids)['last_hidden_state']

            # get the final persona sentence embedding
            # persona_emb_r = self.self_attn(persona_emb)
            batch, n_persona, seq_leng, n_embd = persona_emb.size()
            i = torch.arange(batch * n_persona).reshape(batch * n_persona, 1, 1)
            j = persona_can_idx.reshape(batch * n_persona, 1, 1)
            k = torch.arange(n_embd)
            persona_emb_final = persona_emb.view(batch * n_persona, seq_leng, n_embd)[i, j, k]
            persona_emb_final = persona_emb_final.reshape(batch, n_persona, n_embd)

            # get the final persona sentence embedding
            batch, n_knowledge, seq_leng, n_embd = knowledge_emb.size()
            i = torch.arange(batch * n_knowledge).reshape(batch * n_knowledge, 1, 1)
            j = knowledge_can_idx.reshape(batch * n_knowledge, 1, 1)
            k = torch.arange(n_embd)
            knowledge_emb_final = knowledge_emb.view(batch * n_knowledge, seq_leng, n_embd)[i, j, k]
            knowledge_emb_final = knowledge_emb_final.reshape(batch, n_knowledge, n_embd)

            # reduce persona/knowledge embeddings
            persona_emb_crossed = self.cross_attn_persona(queries=persona_emb_final, keys=knowledge_emb_final,
                                                          values=utterance_emb)
            knowledge_emb_crossed = self.cross_attn_knowledge(queries=knowledge_emb_final, keys=persona_emb_final,
                                                              values=utterance_emb)

            # pick persona
            persona_logits = self.summary_persona(persona_emb_crossed + persona_emb_final).view(batch, -1)
            outputs = (persona_logits,) + outputs
            persona_pred_sigmoid = self.sigmoid(persona_logits)
            persona_pred_sigmoid = (persona_pred_sigmoid > 0.5).float()
            all_persona_pred = []
            selected_persona_idx = list()
            for batch_idx, persona_batch in enumerate(torch.eq(persona_pred_sigmoid, 1)):
                batch_list_idx = list()
                batch_list = list()
                for i, can in enumerate(persona_batch):
                    if can == True:
                        batch_list_idx.append(can)
                        persona_selected_now = persona_input_ids[batch_idx][i]
                        mask_persona = torch.ne(persona_selected_now, padding)
                        persona_selected_now = torch.masked_select(persona_selected_now, mask_persona)
                        batch_list.append(persona_selected_now[:-2])
                all_persona_pred.append(batch_list)
                selected_persona_idx.append(batch_list_idx)

            # pick knowledge
            knowledge_logits = self.summary_knowledge(knowledge_emb_crossed + knowledge_emb_final).view(batch, -1)
            outputs = (knowledge_logits,) + outputs
            knowledge_softmax = self.softmax(knowledge_logits)
            _, k_index_1 = torch.topk(knowledge_softmax, k=1, dim=-1)
            all_knowledge_pred = []
            for batch_i in range(batch):
                knowledge_pred_idx = k_index_1[batch_i]
                knowledge_pred = knowledge_input_ids[batch_i][knowledge_pred_idx]
                mask_knowledge = torch.ne(knowledge_pred, padding)
                knowledge_pred = torch.masked_select(knowledge_pred, mask_knowledge)
                knowledge_pred = knowledge_pred[1:-2]  # delete bos, knowledge_st, eos
                if knowledge_pred.size()[0] > 150:
                    knowledge_pred = knowledge_pred[:150]
                all_knowledge_pred.append(knowledge_pred)

            final_input_list = []
            final_input_tti_list = []
            final_lm_label_list = []
            for batch_i in range(batch):
                only_dial_input_ids_batch = only_dial_input_ids[batch_i]
                only_dial_token_type_ids_batch = only_dial_token_type_ids[batch_i]
                mask_only_dial_input_ids_batch = torch.ne(only_dial_input_ids_batch, padding)
                mask_only_dial_tti_batch = torch.ne(only_dial_token_type_ids_batch, padding)
                only_dial_input_ids_batch = torch.masked_select(only_dial_input_ids_batch,
                                                                mask_only_dial_input_ids_batch)
                only_dial_token_type_ids_batch = torch.masked_select(only_dial_token_type_ids_batch,
                                                                     mask_only_dial_tti_batch)
                if len(all_persona_pred[batch_i]) > 0:
                    concat_persona = torch.cat(all_persona_pred[batch_i], dim=-1)
                    new_persona = torch.cat([persona_tensor, concat_persona], dim=-1)
                    new_persona_tti = torch.tensor([persona] * (new_persona.size()[0])).to(device)
                else:
                    new_persona = None
                    new_persona_tti = None

                new_knowledge = torch.cat([knowledge_tensor, all_knowledge_pred[batch_i]], dim=-1)
                new_knowledge_tti = torch.tensor([knowledge] * (new_knowledge.size()[0])).to(device)

                if new_persona is not None:
                    new_input = torch.cat([bos_tensor, new_knowledge, new_persona, only_dial_input_ids_batch], dim=-1)
                    new_input_tti = torch.cat(
                        [knowledge_tensor, new_knowledge_tti, new_persona_tti, only_dial_token_type_ids_batch], dim=-1)
                else:
                    new_input = torch.cat([bos_tensor, new_knowledge, only_dial_input_ids_batch], dim=-1)
                    new_input_tti = torch.cat([knowledge_tensor, new_knowledge_tti, only_dial_token_type_ids_batch],
                                              dim=-1)

                new_input_size = new_input.size()[0]
                new_lm_label = torch.cat(
                    [torch.tensor([-100] * (new_input_size - (only_dial_input_ids_batch.size()[0]) + 1)).to(device),
                     only_dial_input_ids_batch[1:]], dim=-1)

                assert new_input.size() == new_input_tti.size() == new_lm_label.size()
                if new_input_size < int(self.max_position):
                    padding_size = int(self.max_position) - new_input_size
                    add_padding = torch.tensor([padding] * padding_size).to(device)
                    add_lm_padding = torch.tensor([-100] * padding_size).to(device)
                    final_input = torch.cat([new_input, add_padding], dim=-1)
                    final_tti_input = torch.cat([new_input_tti, add_padding], dim=-1)
                    final_lm_label = torch.cat([new_lm_label, add_lm_padding], dim=-1)
                final_input_list.append(final_input)
                final_input_tti_list.append(final_tti_input)
                final_lm_label_list.append(final_lm_label)
            input_ids = torch.stack(final_input_list)
            token_type_ids = torch.stack(final_input_tti_list)
            lm_labels = torch.stack(final_lm_label_list)

        dynamic_lm_hidden_states = self.transformer(input_ids=input_ids, token_type_ids=token_type_ids)[
            'last_hidden_state']
        if dynamic_lm_hidden_states is not None:
            dynamic_lm_logits = self.lm_head(dynamic_lm_hidden_states)
            outputs = (dynamic_lm_logits,) + outputs

        if persona_grounding is not None:
            loss_fct = BCEWithLogitsLoss()
            persona_loss = loss_fct(persona_logits.view(batch, -1), persona_grounding.type_as(persona_logits))
            outputs = (persona_loss,) + outputs

        if knowledge_grounding is not None:
            loss_fct = CrossEntropyLoss()
            knowledge_loss = loss_fct(knowledge_logits.view(batch, -1), knowledge_grounding)
            outputs = (knowledge_loss,) + outputs

        if training is not True:
            outputs = (lm_labels,) + outputs
            lm_labels = None

        if lm_labels is not None:
            shift_logits = dynamic_lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (lm_loss,) + outputs

        # train: lm_loss, knowledge_loss, persona_loss, dynamic_lm_logits, knowledge_logits, persona_logits
        # valid: lm_label, dynamic_lm_logits, knowledge_logits, persona_logits
        return outputs


class BARTPK_catatn(BartForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        # config.vocab_size = config.vocab_size + 4
        self.model = BartModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.concat_summary = ConcatSummary(emb_dim=config.d_model)
        self.summary = Summary(emb_dim=config.d_model)
        self.attn1 = nn.Linear(config.d_model, 5)
        self.attn2 = nn.Linear(5, config.d_model)  # Selected knowledge 개수만
        self.cross_attn_persona = ConcatCrossAttention()
        self.cross_attn_knowledge = ConcatCrossAttention()
        self.summary_persona = Summary(emb_dim=config.d_model)
        self.summary_knowledge = Summary(emb_dim=config.d_model)
        self.sigmoid = Sigmoid()
        self.softmax = Softmax(dim=-1)
        self.max_position = config.max_position_embeddings
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            input_eos=None,
            only_dial_input_ids=None,
            decoder_input_ids=None,
            persona_input_ids=None,
            knowledge_input_ids=None,
            persona_can_idx=None,
            persona_grounding=None,
            knowledge_can_idx=None,
            knowledge_grounding=None,
            tot_knowledge=None,
            tot_knowledge_eos=None,
            training=None,
            lm_labels=None,
            mc_token_ids=None):

        # machine = 50265
        # human = 50266
        persona = 50267
        knowledge = 50268
        padding = 1
        bos = 0
        eos = 2
        num_chosen_paragraph = 5
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        persona_tensor = torch.tensor([persona]).to(device)
        knowledge_tensor = torch.tensor([knowledge]).to(device)
        bos_tensor = torch.tensor([bos]).to(device)
        eos_tensor = torch.tensor([eos]).to(device)

        outputs = tuple()
        dynamic_lm_logits = None
        persona_logits = None
        knowledge_logits = None

        if input_eos is not None:
            ######################################################################################
            # encode with BART
            utterance_emb = self.model(input_ids=only_dial_input_ids)['last_hidden_state']
            batch, seq_len, embdim = utterance_emb.size()
            num_persona_can = 5
            persona_emb = self.model(input_ids=persona_input_ids.view(batch * num_persona_can, -1))[
                'last_hidden_state'].view(batch, num_persona_can, -1, embdim)
            num_knowledge_can = 10
            knowledge_emb = self.model(input_ids=knowledge_input_ids.view(batch * num_knowledge_can, -1))[
                'last_hidden_state'].view(batch, num_knowledge_can, -1, embdim)

            # get the final persona sentence embedding
            # persona_emb_r = self.self_attn(persona_emb)
            batch, n_persona, seq_leng, n_embd = persona_emb.size()
            i = torch.arange(batch * n_persona).reshape(batch * n_persona, 1, 1)
            j = persona_can_idx.reshape(batch * n_persona, 1, 1)
            k = torch.arange(n_embd)
            persona_emb_final = persona_emb.view(batch * n_persona, seq_leng, n_embd)[i, j, k]
            persona_emb_final = persona_emb_final.reshape(batch, n_persona, n_embd)

            # get the final persona sentence embedding
            batch, n_knowledge, seq_leng, n_embd = knowledge_emb.size()
            i = torch.arange(batch * n_knowledge).reshape(batch * n_knowledge, 1, 1)
            j = knowledge_can_idx.reshape(batch * n_knowledge, 1, 1)
            k = torch.arange(n_embd)
            knowledge_emb_final = knowledge_emb.view(batch * n_knowledge, seq_leng, n_embd)[i, j, k]
            knowledge_emb_final = knowledge_emb_final.reshape(batch, n_knowledge, n_embd)

            # reduce persona/knowledge embeddings
            persona_emb_crossed = self.cross_attn_persona(queries=persona_emb_final, keys=knowledge_emb_final,
                                                          values=utterance_emb)
            knowledge_emb_crossed = self.cross_attn_knowledge(queries=knowledge_emb_final, keys=persona_emb_final,
                                                              values=utterance_emb)

            # pick persona
            persona_logits = self.summary_persona(persona_emb_crossed + persona_emb_final).view(batch, -1)
            outputs = (persona_logits,) + outputs

            persona_pred_sigmoid = self.sigmoid(persona_logits)
            persona_pred_sigmoid = (persona_pred_sigmoid > 0.5).float()
            all_persona_pred = []
            selected_persona_idx = list()
            for batch_idx, persona_batch in enumerate(torch.eq(persona_pred_sigmoid, 1)):
                batch_list_idx = list()
                batch_list = list()
                for i, can in enumerate(persona_batch):
                    if can == True:
                        batch_list_idx.append(can)
                        persona_selected_now = persona_input_ids[batch_idx][i]
                        mask_persona = torch.ne(persona_selected_now, padding)
                        persona_selected_now = torch.masked_select(persona_selected_now, mask_persona)
                        batch_list.append(persona_selected_now[:-2])
                all_persona_pred.append(batch_list)
                selected_persona_idx.append(batch_list_idx)

            # pick knowledge
            knowledge_logits = self.summary_knowledge(knowledge_emb_crossed + knowledge_emb_final).view(batch, -1)
            outputs = (knowledge_logits,) + outputs
            knowledge_softmax = self.softmax(knowledge_logits)
            _, k_index_1 = torch.topk(knowledge_softmax, k=1, dim=-1)
            all_knowledge_pred = []
            for batch_i in range(batch):
                knowledge_pred_idx = k_index_1[batch_i]
                knowledge_pred = knowledge_input_ids[batch_i][knowledge_pred_idx]
                mask_knowledge = torch.ne(knowledge_pred, padding)
                knowledge_pred = torch.masked_select(knowledge_pred, mask_knowledge)
                knowledge_pred = knowledge_pred[1:-2]
                all_knowledge_pred.append(knowledge_pred)  # delete bos, knowledge_st, eos

            # construct final input
            final_input_list = []
            for batch_i in range(batch):
                only_dial_input_ids_batch = only_dial_input_ids[batch_i]
                mask_only_dial_input_ids_batch = torch.ne(only_dial_input_ids_batch, padding)
                only_dial_input_ids_batch = torch.masked_select(only_dial_input_ids_batch,
                                                                mask_only_dial_input_ids_batch)
                if len(all_persona_pred[batch_i]) > 0:
                    concat_persona = torch.cat(all_persona_pred[batch_i], dim=-1)
                    new_persona = torch.cat([persona_tensor, concat_persona], dim=-1)
                else:
                    new_persona = None

                new_knowledge = torch.cat([knowledge_tensor, all_knowledge_pred[batch_i]], dim=-1)

                if new_persona is not None:
                    new_input = torch.cat(
                        [bos_tensor, new_knowledge, new_persona, only_dial_input_ids_batch, eos_tensor], dim=-1)
                else:
                    new_input = torch.cat([bos_tensor, new_knowledge, only_dial_input_ids_batch, eos_tensor], dim=-1)

                new_input_size = new_input.size()[0]

                if new_input_size < int(self.max_position):
                    padding_size = int(self.max_position) - new_input_size
                    add_padding = torch.tensor([padding] * padding_size).to(device)
                    final_input = torch.cat([new_input, add_padding], dim=-1)
                final_input_list.append(final_input)
            input_ids = torch.stack(final_input_list)

        # call BART model
        dynamic_lm_hidden_states = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)[
            'last_hidden_state']

        if dynamic_lm_hidden_states is not None:
            dynamic_lm_logits = self.lm_head(dynamic_lm_hidden_states)
            outputs = (dynamic_lm_logits,) + outputs

        if persona_grounding is not None:
            loss_fct = BCEWithLogitsLoss()
            persona_loss = loss_fct(persona_logits.view(batch, -1), persona_grounding.type_as(persona_logits))
            outputs = (persona_loss,) + outputs

        if knowledge_grounding is not None:
            loss_fct = CrossEntropyLoss()
            knowledge_loss = loss_fct(knowledge_logits.view(batch, -1), knowledge_grounding)
            outputs = (knowledge_loss,) + outputs

        if training is True:
            shift_logits = dynamic_lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (lm_loss,) + outputs

        return outputs  # (lm_loss-training), (knowledge_loss), (persona_loss), dynamic_lm_logits, knowledge_logits, persona_logits, persona_detect_logits, presents, (all hidden_states), (attentions)


class GPT2PK_colbert(GPT2LMHeadModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.concat_summary = ConcatSummary(emb_dim=config.n_embd)

        self.cross_attn_persona = ColBERTSim()
        self.cross_attn_knowledge = ColBERTSim()
        self.colbert_in_reduce = nn.Linear(config.n_embd, int(config.n_embd / 4), bias=True)
        self.colbert_out_reduce = nn.Linear(2, 1, bias=True)
        self.sigmoid = Sigmoid()
        self.softmax = Softmax(dim=-1)
        self.max_position = config.n_positions
        # self.paragraph_sum = nn.Linear(config.n_embd, 1)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            input_eos=None,
            token_type_ids=None,
            only_dial_input_ids=None,
            only_dial_token_type_ids=None,
            persona_input_ids=None,
            knowledge_input_ids=None,
            persona_can_idx=None,
            persona_grounding=None,
            knowledge_can_idx=None,
            knowledge_grounding=None,
            tot_knowledge=None,
            tot_knowledge_token_ids=None,
            tot_knowledge_eos=None,
            training=None,
            mc_token_ids=None):

        persona = 50259
        knowledge = 50260
        padding = 50261
        bos = 50256
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        persona_tensor = torch.tensor([persona]).to(device)
        knowledge_tensor = torch.tensor([knowledge]).to(device)
        bos_tensor = torch.tensor([bos]).to(device)
        outputs = tuple()
        dynamic_lm_logits = None
        persona_logits = None
        knowledge_logits = None
        lm_labels = None

        if input_eos is not None:
            ######################################################################################
            # encode with transformers
            utterance_emb = self.transformer(input_ids=only_dial_input_ids)['last_hidden_state']
            persona_emb = self.transformer(input_ids=persona_input_ids)['last_hidden_state']
            knowledge_emb = self.transformer(input_ids=knowledge_input_ids)['last_hidden_state']

            mask_u = torch.ne(only_dial_input_ids, padding)
            mask_p = torch.ne(persona_input_ids, padding)
            mask_k = torch.ne(knowledge_input_ids, padding)

            batch, _, _ = utterance_emb.size()

            # reduce persona/knowledge embeddings
            u_emb = self.colbert_in_reduce(utterance_emb)
            p_emb = self.colbert_in_reduce(persona_emb)
            k_emb = self.colbert_in_reduce(knowledge_emb)

            p_x_u = self.cross_attn_persona(cand_rep=p_emb, ctxt_rep=u_emb.unsqueeze(1),
                                            mask_cand=mask_p, mask_ctxt=mask_u.unsqueeze(1))
            p_x_k = self.cross_attn_persona(cand_rep=p_emb, ctxt_rep=k_emb,
                                            mask_cand=mask_p, mask_ctxt=mask_k)

            k_x_u = self.cross_attn_persona(cand_rep=k_emb, ctxt_rep=u_emb.unsqueeze(1),
                                            mask_cand=mask_k, mask_ctxt=mask_u.unsqueeze(1))
            k_x_p = self.cross_attn_persona(cand_rep=k_emb, ctxt_rep=p_emb,
                                            mask_cand=mask_k, mask_ctxt=mask_p)

            # pick persona
            p_x = torch.cat([torch.mean(p_x_k, dim=-1, keepdim=True), p_x_u], dim=-1)
            persona_logits = self.colbert_out_reduce(p_x).view(batch, -1)
            outputs = (persona_logits,) + outputs
            persona_pred_sigmoid = self.sigmoid(persona_logits)
            persona_pred_sigmoid = (persona_pred_sigmoid > 0.5).float()
            all_persona_pred = []
            selected_persona_idx = list()
            for batch_idx, persona_batch in enumerate(torch.eq(persona_pred_sigmoid, 1)):
                batch_list_idx = list()
                batch_list = list()
                for i, can in enumerate(persona_batch):
                    if can == True:
                        batch_list_idx.append(can)
                        persona_selected_now = persona_input_ids[batch_idx][i]
                        mask_p = torch.ne(persona_selected_now, padding)
                        persona_selected_now = torch.masked_select(persona_selected_now, mask_p)
                        batch_list.append(persona_selected_now[:-2])
                all_persona_pred.append(batch_list)
                selected_persona_idx.append(batch_list_idx)

            # pick knowledge
            k_x = torch.cat([torch.mean(k_x_p, dim=-1, keepdim=True), k_x_u], dim=-1)
            knowledge_logits = self.colbert_out_reduce(k_x).view(batch, -1)
            outputs = (knowledge_logits,) + outputs
            knowledge_softmax = self.softmax(knowledge_logits)
            _, k_index_1 = torch.topk(knowledge_softmax, k=1, dim=-1)
            all_knowledge_pred = []
            for batch_i in range(batch):
                knowledge_pred_idx = k_index_1[batch_i]
                knowledge_pred = knowledge_input_ids[batch_i][knowledge_pred_idx]
                mask_k = torch.ne(knowledge_pred, padding)
                knowledge_pred = torch.masked_select(knowledge_pred, mask_k)
                knowledge_pred = knowledge_pred[1:-2]  # delete bos, knowledge_st, eos
                if knowledge_pred.size()[0] > 150:
                    knowledge_pred = knowledge_pred[:150]
                all_knowledge_pred.append(knowledge_pred)

            final_input_list = []
            final_input_tti_list = []
            final_lm_label_list = []
            for batch_i in range(batch):
                only_dial_input_ids_batch = only_dial_input_ids[batch_i]
                only_dial_token_type_ids_batch = only_dial_token_type_ids[batch_i]
                mask_only_dial_input_ids_batch = torch.ne(only_dial_input_ids_batch, padding)
                mask_only_dial_tti_batch = torch.ne(only_dial_token_type_ids_batch, padding)
                only_dial_input_ids_batch = torch.masked_select(only_dial_input_ids_batch,
                                                                mask_only_dial_input_ids_batch)
                only_dial_token_type_ids_batch = torch.masked_select(only_dial_token_type_ids_batch,
                                                                     mask_only_dial_tti_batch)
                if len(all_persona_pred[batch_i]) > 0:
                    concat_persona = torch.cat(all_persona_pred[batch_i], dim=-1)
                    new_persona = torch.cat([persona_tensor, concat_persona], dim=-1)
                    new_persona_tti = torch.tensor([persona] * (new_persona.size()[0])).to(device)
                else:
                    new_persona = None
                    new_persona_tti = None

                new_knowledge = torch.cat([knowledge_tensor, all_knowledge_pred[batch_i]], dim=-1)
                new_knowledge_tti = torch.tensor([knowledge] * (new_knowledge.size()[0])).to(device)

                if new_persona is not None:
                    new_input = torch.cat([bos_tensor, new_knowledge, new_persona, only_dial_input_ids_batch], dim=-1)
                    new_input_tti = torch.cat(
                        [knowledge_tensor, new_knowledge_tti, new_persona_tti, only_dial_token_type_ids_batch], dim=-1)
                else:
                    new_input = torch.cat([bos_tensor, new_knowledge, only_dial_input_ids_batch], dim=-1)
                    new_input_tti = torch.cat([knowledge_tensor, new_knowledge_tti, only_dial_token_type_ids_batch],
                                              dim=-1)

                new_input_size = new_input.size()[0]
                new_lm_label = torch.cat(
                    [torch.tensor([-100] * (new_input_size - (only_dial_input_ids_batch.size()[0]) + 1)).to(device),
                     only_dial_input_ids_batch[1:]], dim=-1)

                assert new_input.size() == new_input_tti.size() == new_lm_label.size()
                if new_input_size < int(self.max_position):
                    padding_size = int(self.max_position) - new_input_size
                    add_padding = torch.tensor([padding] * padding_size).to(device)
                    add_lm_padding = torch.tensor([-100] * padding_size).to(device)
                    final_input = torch.cat([new_input, add_padding], dim=-1)
                    final_tti_input = torch.cat([new_input_tti, add_padding], dim=-1)
                    final_lm_label = torch.cat([new_lm_label, add_lm_padding], dim=-1)
                final_input_list.append(final_input)
                final_input_tti_list.append(final_tti_input)
                final_lm_label_list.append(final_lm_label)
            input_ids = torch.stack(final_input_list)
            token_type_ids = torch.stack(final_input_tti_list)
            lm_labels = torch.stack(final_lm_label_list)

        dynamic_lm_hidden_states = self.transformer(input_ids=input_ids, token_type_ids=token_type_ids)[
            'last_hidden_state']
        if dynamic_lm_hidden_states is not None:
            dynamic_lm_logits = self.lm_head(dynamic_lm_hidden_states)
            outputs = (dynamic_lm_logits,) + outputs

        if persona_grounding is not None:
            loss_fct = BCEWithLogitsLoss()
            persona_loss = loss_fct(persona_logits.view(batch, -1), persona_grounding.type_as(persona_logits))
            outputs = (persona_loss,) + outputs

        if knowledge_grounding is not None:
            loss_fct = CrossEntropyLoss()
            knowledge_loss = loss_fct(knowledge_logits.view(batch, -1), knowledge_grounding)
            outputs = (knowledge_loss,) + outputs

        if training is not True:
            outputs = (lm_labels,) + outputs
            lm_labels = None

        if lm_labels is not None:
            shift_logits = dynamic_lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (lm_loss,) + outputs

        # train: lm_loss, knowledge_loss, persona_loss, dynamic_lm_logits, knowledge_logits, persona_logits
        # valid: lm_label, dynamic_lm_logits, knowledge_logits, persona_logits
        return outputs

class BARTPK_colbert(BartForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        # config.vocab_size = config.vocab_size + 4
        self.model = BartModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.concat_summary = ConcatSummary(emb_dim=config.d_model)
        self.summary = Summary(emb_dim=config.d_model)
        self.cross_attn_persona = ColBERTSim()
        self.cross_attn_knowledge = ColBERTSim()
        self.colbert_in_reduce = nn.Linear(config.d_model, int(config.d_model / 4), bias=True)
        self.colbert_out_reduce = nn.Linear(2, 1, bias=True)
        self.sigmoid = Sigmoid()
        self.softmax = Softmax(dim=-1)
        self.max_position = config.max_position_embeddings
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            input_eos=None,
            only_dial_input_ids=None,
            decoder_input_ids=None,
            persona_input_ids=None,
            knowledge_input_ids=None,
            persona_can_idx=None,
            persona_grounding=None,
            knowledge_can_idx=None,
            knowledge_grounding=None,
            tot_knowledge=None,
            tot_knowledge_eos=None,
            training=None,
            lm_labels=None,
            mc_token_ids=None):

        # machine = 50265
        # human = 50266
        persona = 50267
        knowledge = 50268
        padding = 1
        bos = 0
        eos = 2
        num_chosen_paragraph = 5
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        persona_tensor = torch.tensor([persona]).to(device)
        knowledge_tensor = torch.tensor([knowledge]).to(device)
        bos_tensor = torch.tensor([bos]).to(device)
        eos_tensor = torch.tensor([eos]).to(device)

        outputs = tuple()
        dynamic_lm_logits = None
        persona_logits = None
        knowledge_logits = None

        if input_eos is not None:
            ######################################################################################
            # encode with BART
            utterance_emb = self.model(input_ids=only_dial_input_ids)['last_hidden_state']
            batch, seq_len, embdim = utterance_emb.size()
            num_persona_can = 5
            persona_emb = self.model(input_ids=persona_input_ids.view(batch * num_persona_can, -1))[
                'last_hidden_state'].view(batch, num_persona_can, -1, embdim)
            num_knowledge_can = 10
            knowledge_emb = self.model(input_ids=knowledge_input_ids.view(batch * num_knowledge_can, -1))[
                'last_hidden_state'].view(batch, num_knowledge_can, -1, embdim)

            mask_u = torch.ne(only_dial_input_ids, padding)
            mask_p = torch.ne(persona_input_ids, padding)
            mask_k = torch.ne(knowledge_input_ids, padding)

            batch, _, _ = utterance_emb.size()

            # reduce persona/knowledge embeddings
            u_emb = self.colbert_in_reduce(utterance_emb)
            p_emb = self.colbert_in_reduce(persona_emb)
            k_emb = self.colbert_in_reduce(knowledge_emb)

            p_x_u = self.cross_attn_persona(cand_rep=p_emb, ctxt_rep=u_emb.unsqueeze(1),
                                            mask_cand=mask_p, mask_ctxt=mask_u.unsqueeze(1))
            p_x_k = self.cross_attn_persona(cand_rep=p_emb, ctxt_rep=k_emb,
                                            mask_cand=mask_p, mask_ctxt=mask_k)

            k_x_u = self.cross_attn_persona(cand_rep=k_emb, ctxt_rep=u_emb.unsqueeze(1),
                                            mask_cand=mask_k, mask_ctxt=mask_u.unsqueeze(1))
            k_x_p = self.cross_attn_persona(cand_rep=k_emb, ctxt_rep=p_emb,
                                            mask_cand=mask_k, mask_ctxt=mask_p)

            # pick persona
            p_x = torch.cat([torch.mean(p_x_k, dim=-1, keepdim=True), p_x_u], dim=-1)
            persona_logits = self.colbert_out_reduce(p_x).view(batch, -1)
            outputs = (persona_logits,) + outputs

            persona_pred_sigmoid = self.sigmoid(persona_logits)
            persona_pred_sigmoid = (persona_pred_sigmoid > 0.5).float()
            all_persona_pred = []
            selected_persona_idx = list()
            for batch_idx, persona_batch in enumerate(torch.eq(persona_pred_sigmoid, 1)):
                batch_list_idx = list()
                batch_list = list()
                for i, can in enumerate(persona_batch):
                    if can == True:
                        batch_list_idx.append(can)
                        persona_selected_now = persona_input_ids[batch_idx][i]
                        mask_persona = torch.ne(persona_selected_now, padding)
                        persona_selected_now = torch.masked_select(persona_selected_now, mask_persona)
                        batch_list.append(persona_selected_now[:-2])
                all_persona_pred.append(batch_list)
                selected_persona_idx.append(batch_list_idx)

            # pick knowledge
            k_x = torch.cat([torch.mean(k_x_p, dim=-1, keepdim=True), k_x_u], dim=-1)
            knowledge_logits = self.colbert_out_reduce(k_x).view(batch, -1)
            outputs = (knowledge_logits,) + outputs
            knowledge_softmax = self.softmax(knowledge_logits)
            _, k_index_1 = torch.topk(knowledge_softmax, k=1, dim=-1)
            all_knowledge_pred = []
            for batch_i in range(batch):
                knowledge_pred_idx = k_index_1[batch_i]
                knowledge_pred = knowledge_input_ids[batch_i][knowledge_pred_idx]
                mask_knowledge = torch.ne(knowledge_pred, padding)
                knowledge_pred = torch.masked_select(knowledge_pred, mask_knowledge)
                knowledge_pred = knowledge_pred[1:-2]
                all_knowledge_pred.append(knowledge_pred)  # delete bos, knowledge_st, eos

            # construct final input
            final_input_list = []
            for batch_i in range(batch):
                only_dial_input_ids_batch = only_dial_input_ids[batch_i]
                mask_only_dial_input_ids_batch = torch.ne(only_dial_input_ids_batch, padding)
                only_dial_input_ids_batch = torch.masked_select(only_dial_input_ids_batch,
                                                                mask_only_dial_input_ids_batch)
                if len(all_persona_pred[batch_i]) > 0:
                    concat_persona = torch.cat(all_persona_pred[batch_i], dim=-1)
                    new_persona = torch.cat([persona_tensor, concat_persona], dim=-1)
                else:
                    new_persona = None

                new_knowledge = torch.cat([knowledge_tensor, all_knowledge_pred[batch_i]], dim=-1)

                if new_persona is not None:
                    new_input = torch.cat(
                        [bos_tensor, new_knowledge, new_persona, only_dial_input_ids_batch, eos_tensor], dim=-1)
                else:
                    new_input = torch.cat([bos_tensor, new_knowledge, only_dial_input_ids_batch, eos_tensor], dim=-1)

                new_input_size = new_input.size()[0]

                if new_input_size < int(self.max_position):
                    padding_size = int(self.max_position) - new_input_size
                    add_padding = torch.tensor([padding] * padding_size).to(device)
                    final_input = torch.cat([new_input, add_padding], dim=-1)
                final_input_list.append(final_input)
            input_ids = torch.stack(final_input_list)

        # call BART model
        dynamic_lm_hidden_states = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)[
            'last_hidden_state']

        if dynamic_lm_hidden_states is not None:
            dynamic_lm_logits = self.lm_head(dynamic_lm_hidden_states)
            outputs = (dynamic_lm_logits,) + outputs

        if persona_grounding is not None:
            loss_fct = BCEWithLogitsLoss()
            persona_loss = loss_fct(persona_logits.view(batch, -1), persona_grounding.type_as(persona_logits))
            outputs = (persona_loss,) + outputs

        if knowledge_grounding is not None:
            loss_fct = CrossEntropyLoss()
            knowledge_loss = loss_fct(knowledge_logits.view(batch, -1), knowledge_grounding)
            outputs = (knowledge_loss,) + outputs

        if training is True:
            shift_logits = dynamic_lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (lm_loss,) + outputs

        return outputs  # (lm_loss-training), (knowledge_loss), (persona_loss), dynamic_lm_logits, knowledge_logits, persona_logits, persona_detect_logits, presents, (all hidden_states), (attentions)