import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from components.dataset import Batch
from grammar.transition_system import ApplyRuleAction, GenTokenAction, PlaceHolderAction, ActionTree
from grammar.hypothesis import Hypothesis
import numpy as np
import os
from common.config import update_args

from grammar.streg.streg_transition_system import partial_asdl_ast_to_streg_ast, batch_preverify_regex_with_exs, asdl_ast_to_streg_ast, is_equal_ast, is_partial_ast

class CompositeTypeModule(nn.Module):
    def __init__(self, args, type, productions, input_size=0):
        super().__init__()
        self.type = type
        self.productions = productions
        h_size = input_size if input_size > 0 else 2 * args.enc_hid_size + args.field_emb_size
        self.w = nn.Linear(h_size, len(productions))
        # self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        return self.w(x)

    # x b * h
    def score(self, x, contexts, io_contexts=None):
        if io_contexts is None:
            x = torch.cat([x, contexts], dim=1)
        else:
            x = torch.cat([x, contexts, io_contexts], dim=1)

        return F.log_softmax(self.w(x),1)

    def probs(self, x, contexts):
        x = torch.cat([x, contexts], dim=1)
        return F.softmax(self.w(x),1)

class ConstructorTypeModule(nn.Module):
    def __init__(self,  args, production, input_size=0):
        super().__init__()
        self.production = production
        self.n_field = len(production.constructor.fields)
        self.field_embeddings = nn.Embedding(len(production.constructor.fields), args.field_emb_size)
        h_size = input_size if input_size > 0 else 2 * args.enc_hid_size + args.field_emb_size
        self.w = nn.Linear(h_size, args.enc_hid_size)
        self.dropout = nn.Dropout(args.dropout)
    
    def update(self, v_lstm, v_state, contexts, io_contexts=None):
        # v_state, h_n, c_n where 1 * b * h
        # input: seq_len, batch, input_size
        # h_0 of shape (1, batch, hidden_size)
        # v_lstm(, v_state)
        inputs = self.field_embeddings.weight
        inputs = self.dropout(inputs)
        contexts = contexts.expand([self.n_field, -1])
        if io_contexts is None:
            inputs = self.w(torch.cat([inputs, contexts], dim=1)).unsqueeze(0)
        else:
            io_contexts = io_contexts.expand([self.n_field, -1])
            inputs = self.w(torch.cat([inputs, contexts, io_contexts], dim=1)).unsqueeze(0)
        v_state = (v_state[0].expand(self.n_field, -1).unsqueeze(0), v_state[1].expand(self.n_field, -1).unsqueeze(0))
        _, outputs = v_lstm(inputs, v_state)

        hidden_states = outputs[0].unbind(1)
        cell_states = outputs[1].unbind(1)

        return list(zip(hidden_states, cell_states))

class PrimitiveTypeModule(nn.Module):
    def __init__(self, args, type, vocab, input_size=0):
        super().__init__()
        self.type = type
        self.vocab = vocab
        h_size = input_size if input_size > 0 else 2 * args.enc_hid_size + args.field_emb_size
        self.w = nn.Linear(h_size, len(vocab))

    def forward(self, x):
        return self.w(x)
    # need a score

    # x b * h
    def score(self, x, contexts, io_contexts=None):
        if io_contexts is None:
            x = torch.cat([x, contexts], dim=1)
        else:
            x = torch.cat([x, contexts, io_contexts], dim=1)
        return F.log_softmax(self.w(x),1)

    def probs(self,x, contexts):
        x = torch.cat([x, contexts], dim=1)
        return F.softmax(self.w(x),1)

class ASNParser(nn.Module):
    def __init__(self, args, transition_system, vocab):
        super().__init__()

        # encoder
        self.args = args
        self.src_embedding = EmbeddingLayer(args.src_emb_size, vocab.src_vocab.size(), args.dropout)
        self.encoder = RNNEncoder(args.src_emb_size, args.enc_hid_size, args.dropout, True)
        self.transition_system = transition_system
        self.vocab = vocab
        grammar = transition_system.grammar
        self.grammar = grammar
        # init
        comp_type_modules = []
        for dsl_type in grammar.composite_types:
            comp_type_modules.append((dsl_type.name,
                                      CompositeTypeModule(args, dsl_type, grammar.get_prods_by_type(dsl_type))))
        self.comp_type_dict = nn.ModuleDict(comp_type_modules)

        # init
        cnstr_type_modules = []
        for prod in grammar.productions:
            cnstr_type_modules.append((prod.constructor.name,
                                       ConstructorTypeModule(args, prod)))
        self.const_type_dict = nn.ModuleDict(cnstr_type_modules)

        prim_type_modules = []
        for prim_type in grammar.primitive_types:
            prim_type_modules.append((prim_type.name,
                                      PrimitiveTypeModule(args, prim_type, vocab.primitive_vocabs[prim_type])))
        self.prim_type_dict = nn.ModuleDict(prim_type_modules)

        self.v_lstm = nn.LSTM(args.enc_hid_size, args.enc_hid_size)
        self.attn = LuongAttention(args.enc_hid_size, 2 * args.enc_hid_size)
        self.dropout = nn.Dropout(args.dropout)

    def score(self, examples):
        # for ex in examples:
        scores = [self._score(ex) for ex in examples]
        # print(scores)
        return torch.stack(scores)
    

    def _score(self, ex):
        batch = Batch([ex], self.grammar, self.vocab)
        context_vecs, encoder_outputs = self.encode(batch)
        init_state = encoder_outputs

        return self._score_node(self.grammar.root_type, init_state, ex.tgt_actions, context_vecs, batch.sent_masks)

    def encode(self, batch):
        sent_lens = batch.sent_lens
        # sent
        sent_embedding =  self.src_embedding(batch.sents)
        context_vecs, final_state = self.encoder(sent_embedding, sent_lens)

        # L * b * hidden,  
        # print(context_vecs.size(), final_state[0].size(), final_state[1].size())
        return context_vecs, final_state

    def _score_node(self, node_type, v_state, action_node, context_vecs, context_masks):
        v_output = self.dropout(v_state[0])
        contexts = self.attn(v_output.unsqueeze(0), context_vecs).squeeze(0)
        if node_type.is_primitive_type():
            module = self.prim_type_dict[node_type.name]
            # scores = mask * module()
            scores = module.score(v_output, contexts)
            # scores =  [tgt_action_tree.action.choice_idx]
            # b * choice
            score = -1 * scores.view([-1])[action_node.action.choice_index]
            # print("Primitive", score)
            return score

        
        cnstr = action_node.action.choice.constructor
        comp_module = self.comp_type_dict[node_type.name]
        scores = comp_module.score(v_output, contexts)
        score = -1 * scores.view([-1])[action_node.action.choice_index]
        # print("Apply", score)

        # pass through
        cnstr_module = self.const_type_dict[cnstr.name]
        # cnstr_results = const_module.iup()
        # next_states = self.v_lstm( [1 * 1 * x], v_state)
        cnstr_results = cnstr_module.update(self.v_lstm, v_state, contexts)
        for next_field, next_state, next_action in zip(cnstr.fields, cnstr_results, action_node.fields):
            score += self._score_node(next_field.type, next_state, next_action, context_vecs, context_masks)
        return score

    def naive_parse(self, ex):
        batch = Batch([ex], self.grammar, self.vocab, train=False)        
        context_vecs, encoder_outputs = self.encode(batch)
        init_state = encoder_outputs

        action_tree = self._naive_parse(self.grammar.root_type, init_state, context_vecs, batch.sent_masks, 1)

        return self.transition_system.build_ast_from_actions(action_tree)

    def _naive_parse(self, node_type, v_state, context_vecs, context_masks, depth):

        # v_state = v_state.torch.unsqueeze(0)

        # tgt_production if production needed
        # tgt_production = tgt

        # else token needed
        # tgt_token = tgt
        contexts = self.attn(v_state[0].unsqueeze(0), context_vecs).squeeze(0)
        if depth > 9:
            return ActionTree(PlaceHolderAction(node_type))

        if node_type.is_primitive_type():
            module = self.prim_type_dict[node_type.name]
            # scores = mask * module()
            scores = module.score(v_state[0], contexts).cpu().numpy().flatten()
            # scores =  [tgt_action_tree.action.choice_idx]
            # b * choice
            # score = -1 * scores.view([-1])[action_node.action.choice_index]
            choice_idx = np.argmax(scores)
            return ActionTree(GenTokenAction(node_type, module.vocab.get_word(choice_idx)))

        
        comp_module = self.comp_type_dict[node_type.name]
        scores = comp_module.score(v_state[0], contexts).cpu().numpy().flatten()
        choice_idx = np.argmax(scores)
        production = comp_module.productions[choice_idx]

        action = ApplyRuleAction(node_type, production)
        cnstr = production.constructor

        # pass through
        cnstr_module = self.const_type_dict[cnstr.name]
        # cnstr_results = const_module.iup()
        # next_states = self.v_lstm( [1 * 1 * x], v_state)
        cnstr_results = cnstr_module.update(self.v_lstm, v_state, contexts)
        action_fields = [self._naive_parse(next_field.type, next_state, context_vecs, context_masks, depth+1) for next_field, next_state in zip(cnstr.fields, cnstr_results)]

        return ActionTree(action, action_fields)


    def ex_guided_parse(self, ex, cache=None):
        return self.ex_guided_parse_and_track_budget(ex, cache=cache)
    
    def precompute_scores_for_holes(self, hyp, context_vecs, context_masks):
        # v_state = v_state.torch.unsqueeze(0)

        # tgt_production if production needed
        # tgt_production = tgt

        # else token needed
        # tgt_token = tgt
        
        nodes_to_proc = hyp.get_unprocessed_nodes()

        for pending_node in nodes_to_proc:
            v_state = pending_node.action.v_state

            contexts = self.attn(v_state[0].unsqueeze(0), context_vecs).squeeze(0)
            node_type = pending_node.action.type
            
            # smooth_factor = 
            if self.args.smooth == 'const':
                smooth_factor = self.args.smooth_alpha 
            elif self.args.smooth == 'prop':
                smooth_factor = pending_node.depth * self.args.smooth_alpha
            else:
                smooth_factor = 0.
                
            if node_type.is_primitive_type():
                module = self.prim_type_dict[node_type.name]
                # scores = mask * module()
                scores = module.score(v_state[0], contexts)
                scores = scores.squeeze(0)
                pending_node.action.cache_scores(scores, module.vocab, smooth_factor=smooth_factor)
            else:
                comp_module = self.comp_type_dict[node_type.name]
                scores = comp_module.score(v_state[0], contexts)
                scores = scores.squeeze(0)
                pending_node.action.cache_scores(scores, comp_module.productions, smooth_factor=smooth_factor)

    
        hyp.update_priority_scores()

    def ex_guided_parse_with_prioritized_order(self, ex, cache=None):
        batch = Batch([ex], self.grammar, self.vocab, train=False)        
        context_vecs, encoder_outputs = self.encode(batch)
        init_state = encoder_outputs
        
        completed_hyps = []
        # cur_beam = [Hypothesis.init_hypothesis(self.grammar.root_type, init_state)]
        init_hyp = Hypothesis.init_hypothesis(self.grammar.root_type, init_state, order=self.args.search_order)
        self.precompute_scores_for_holes(init_hyp, context_vecs, batch.sent_masks)
        cur_beam = [init_hyp]
        num_executed = 0
        for ts in range(self.args.max_decode_step):
            hyp_pools = []
            for hyp in cur_beam:
                continuations = self.continuations_of_hyp_with_cache(hyp, context_vecs, batch.sent_masks)
                hyp_pools.extend(continuations)
            checkable_pool = []
            non_checkable_pool = []
            for hyp in hyp_pools:
                # asdl_ast = self.transition_system.build_ast_from_actions(hyp.action_tree)
                is_checkable, partial_ast = partial_asdl_ast_to_streg_ast(self.transition_system.build_ast_from_actions(hyp.action_tree))
                if is_checkable:
                    checkable_pool.append((hyp, partial_ast))
                else:
                    non_checkable_pool.append((hyp, partial_ast))

            hyp_pools = [x[0] for x in non_checkable_pool]
            if checkable_pool:
                batch_results = batch_preverify_regex_with_exs([x[1] for x in checkable_pool], ex, cache=cache)
                hyp_pools += [x[0] for (x,y) in zip(checkable_pool, batch_results) if y]
            hyp_pools.sort(key=lambda x: x.score, reverse=True)
            num_slots = self.args.beam_size - len(completed_hyps)

            cur_beam = []
            for hyp_i, hyp  in enumerate(hyp_pools[:num_slots]):
                if hyp.is_complete():
                    completed_hyps.append(hyp)
                else:
                    self.precompute_scores_for_holes(hyp, context_vecs, batch.sent_masks)
                    cur_beam.append(hyp)
            if not cur_beam:
                break
        print(len(completed_hyps), 'num exec', num_executed)
        completed_hyps.sort(key=lambda x: x.score, reverse=True)
        return completed_hyps


    def ex_guided_parse_and_track_budget(self, ex, cache=None):
        batch = Batch([ex], self.grammar, self.vocab, train=False)        
        context_vecs, encoder_outputs = self.encode(batch)
        init_state = encoder_outputs
        
        completed_hyps = []
        # cur_beam = [Hypothesis.init_hypothesis(self.grammar.root_type, init_state)]
        init_hyp = Hypothesis.init_hypothesis(self.grammar.root_type, init_state, order=self.args.search_order)
        self.precompute_scores_for_holes(init_hyp, context_vecs, batch.sent_masks)
        cur_beam = [init_hyp]
        num_executed = 0

        for ts in range(self.args.max_decode_step):
            hyp_pools = []
            for hyp in cur_beam:
                continuations = self.continuations_of_hyp_with_cache(hyp, context_vecs, batch.sent_masks)
                hyp_pools.extend(continuations)
            checkable_pool = []
            non_checkable_pool = []
            for hyp in hyp_pools:
                # asdl_ast = self.transition_system.build_ast_from_actions(hyp.action_tree)
                is_checkable, partial_ast = partial_asdl_ast_to_streg_ast(self.transition_system.build_ast_from_actions(hyp.action_tree))
                if is_checkable:
                    checkable_pool.append((hyp, partial_ast))
                else:
                    non_checkable_pool.append((hyp, partial_ast))

            hyp_pools = [x[0] for x in non_checkable_pool]
            all_scores_this_time = [x[0].score for x in checkable_pool] + [x[0].score for x in non_checkable_pool]
            all_scores_this_time.sort(reverse=True)

            if checkable_pool:
                batch_results = batch_preverify_regex_with_exs([x[1] for x in checkable_pool], ex, cache=cache)
                hyp_pools += [x[0] for (x,y) in zip(checkable_pool, batch_results) if y]
            hyp_pools.sort(key=lambda x: x.score, reverse=True)
            num_slots = self.args.beam_size - len(completed_hyps)

            cur_beam = []
            cur_executed = num_executed
            for hyp_i, hyp  in enumerate(hyp_pools[:num_slots]):
                cur_score = hyp.score
                exec_gain = sum([x >= cur_score for x in all_scores_this_time])
                num_executed = cur_executed + exec_gain
                if hyp.is_complete():
                    hyp.budget_when_reached = num_executed
                    completed_hyps.append(hyp)
                else:
                    self.precompute_scores_for_holes(hyp, context_vecs, batch.sent_masks)
                    cur_beam.append(hyp)
            if len(hyp_pools) < num_slots:
                num_executed = cur_executed + len(all_scores_this_time)
            if not cur_beam:
                break
        print('Final Num', num_executed)
        completed_hyps.sort(key=lambda x: x.score, reverse=True)
        return completed_hyps, num_executed

    def parse(self, ex):
        batch = Batch([ex], self.grammar, self.vocab, train=False)        
        context_vecs, encoder_outputs = self.encode(batch)
        init_state = encoder_outputs

        
        completed_hyps = []
        init_hyp = Hypothesis.init_hypothesis(self.grammar.root_type, init_state, order=self.args.search_order)
        self.precompute_scores_for_holes(init_hyp, context_vecs, batch.sent_masks)
        cur_beam = [init_hyp]
        for ts in range(self.args.max_decode_step):
            hyp_pools = []
            for hyp in cur_beam:
                continuations = self.continuations_of_hyp_with_cache(hyp, context_vecs, batch.sent_masks)
                hyp_pools.extend(continuations)
            
            hyp_pools.sort(key=lambda x: x.score, reverse=True)
            # next_beam = next_beam[:self.args.beam_size]
            
            num_slots = self.args.beam_size - len(completed_hyps)

            cur_beam = []
            for hyp_i, hyp  in enumerate(hyp_pools[:num_slots]):
                if hyp.is_complete():
                    completed_hyps.append(hyp)
                else:
                    self.precompute_scores_for_holes(hyp, context_vecs, batch.sent_masks)
                    cur_beam.append(hyp)
            
            if not cur_beam:
                break

        completed_hyps.sort(key=lambda x: x.score, reverse=True)
        return completed_hyps

    def continuations_of_hyp_with_cache(self, hyp, context_vecs, context_masks):        
        pending_node = hyp.get_pending_node()
        v_state = pending_node.action.v_state
        
        contexts = self.attn(v_state[0].unsqueeze(0), context_vecs).squeeze(0)

        node_type = pending_node.action.type
        scores = pending_node.action.candidate_scores.tolist()
        if node_type.is_primitive_type():
            module = self.prim_type_dict[node_type.name]
            # scores = mask * module()
            # scores =  [tgt_action_tree.action.choice_idx]
            # b * choice
            # score = -1 * scores.view([-1])[action_node.action.choice_index]
            # choice_idx = np.argmax(scores)
            continuous = []
            for choice_idx, score in enumerate(scores):
                continuous.append(hyp.copy_and_apply_action(GenTokenAction(node_type, module.vocab.get_word(choice_idx)), score))
                # return ActionTree()
            return continuous

        comp_module = self.comp_type_dict[node_type.name]
        continuous = []
        for choice_idx, score in enumerate(scores):
            production = comp_module.productions[choice_idx]
            action = ApplyRuleAction(node_type, production)
            cnstr = production.constructor
            # pass through
            cnstr_module = self.const_type_dict[cnstr.name]
            # cnstr_results = const_module.iup()
            # next_states = self.v_lstm( [1 * 1 * x], v_state)
            cnstr_results = cnstr_module.update(self.v_lstm, v_state, contexts)
            continuous.append(hyp.copy_and_apply_action(action, score, cnstr_results))
        return continuous

    # for synthesizing
    def initialize_hyp_for_synthesizing(self, ex):
        batch = Batch([ex], self.grammar, self.vocab, train=False)        
        context_vecs, encoder_outputs = self.encode(batch)
        init_state = encoder_outputs
        init_hyp = Hypothesis.init_hypothesis(self.grammar.root_type, init_state, order=self.args.search_order)
        self.precompute_scores_for_holes(init_hyp, context_vecs, batch.sent_masks)
        return init_hyp, (context_vecs, batch.sent_masks)

    def continuations_of_hyp_for_synthesizer(self, hyp, parser_state):
        if (hyp.t + 1) >= self.args.max_decode_step:
            return []
        continuations = self.continuations_of_hyp_with_cache(hyp, parser_state[0], parser_state[1])
        for c in continuations:
            self.precompute_scores_for_holes(c, parser_state[0], parser_state[1])
        return continuations

    def save(self, filename):
        dir_name = os.path.dirname(filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'transition_system': self.transition_system,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, filename)

    @classmethod
    def load(cls, model_path, ex_args=None, cuda=False):
        params = torch.load(model_path)
        vocab = params['vocab']
        transition_system = params['transition_system']
        saved_args = params['args']
        # update saved args
        saved_state = params['state_dict']
        saved_args.cuda = cuda
        if ex_args:
            update_args(saved_args, ex_args)
        parser = cls(saved_args, transition_system, vocab)
        parser.load_state_dict(saved_state)
        
        # setattr(saved_args, )
        if cuda: parser = parser.cuda()
        parser.eval()

        return parser

class EmbeddingLayer(nn.Module):
    def __init__(self, embedding_dim, full_dict_size, embedding_dropout_rate):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(full_dict_size, embedding_dim)
        self.dropout = nn.Dropout(embedding_dropout_rate)

        nn.init.uniform_(self.embedding.weight, -1, 1)

    def forward(self, input):
        embedded_words = self.embedding(input)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings

class RNNEncoder(nn.Module):
    # Parameters: input size (should match embedding layer), hidden size for the LSTM, dropout rate for the RNN,
    # and a boolean flag for whether or not we're using a bidirectional encoder
    def __init__(self, input_size, hidden_size, dropout, bidirect, reduce=True):
        super(RNNEncoder, self).__init__()
        self.bidirect = bidirect
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reduce = reduce
        if self.reduce:
            self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
            self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=self.bidirect)
        self.init_weight()
        self.dropout = nn.Dropout(dropout)

    # Initializes weight matrices using Xavier initialization
    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        if self.bidirect:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        if self.bidirect:
            nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)

    def get_output_size(self):
        return self.hidden_size * 2 if self.bidirect else self.hidden_size

    # embedded_words should be a [batch size x sent len x input dim] tensor
    # input_lens is a tensor containing the length of each input sentence
    # Returns output (each word's representation), context_mask (a mask of 0s and 1s
    # reflecting where the model's output should be considered), and h_t, a *tuple* containing
    # the final states h and c from the encoder for each sentence.
    def forward(self, embedded_words, input_lens):
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        packed_embedding = nn.utils.rnn.pack_padded_sequence(
            embedded_words, input_lens, batch_first=True)
        # Runs the RNN over each sequence. Returns output at each position as well as the last vectors of the RNN
        # state for each sentence (first/last vectors for bidirectional)
        output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors
        output, _ = nn.utils.rnn.pad_packed_sequence(output)

        # Grabs the encoded representations out of hn, which is a weird tuple thing.
        # Note: if you want multiple LSTM layers, you'll need to change this to consult the penultimate layer
        # or gather representations from all layers.
        if self.bidirect:
            if self.reduce:
                h, c = hn[0], hn[1]
                # Grab the representations from forward and backward LSTMs
                h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat(
                    (c[0], c[1]), dim=1)
                # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
                # as the hidden size in the encoder
                new_h = self.reduce_h_W(h_)
                new_c = self.reduce_c_W(c_)
                h_t = (new_h, new_c)
            else:
                # h, c = hn[0][0], hn[1][0]
                h, c = hn[0], hn[1]
                # Grab the representations from forward and backward LSTMs
                h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
                h_t = (h_, c_)
        else:
            h, c = hn[0][0], hn[1][0]
            h_t = (h, c)
        # print(max_length, output.size(), h_t[0].size(), h_t[1].size())

        output = self.dropout(output)
        return (output, h_t)


class LuongAttention(nn.Module):

    def __init__(self, hidden_size, context_size=None):
        super(LuongAttention, self).__init__()
        self.hidden_size = hidden_size
        self.context_size = hidden_size if context_size is None else context_size
        self.attn = torch.nn.Linear(self.context_size, self.hidden_size)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.attn.weight, gain=1)
        nn.init.constant_(self.attn.bias, 0)

    # input query: q * batch * hidden, contexts: c * batch * hidden
    # output: batch * len * q * c
    def forward(self, query, context, inf_mask=None, requires_weight=False):
        # Calculate the attention weights (energies) based on the given method
        query = query.transpose(0, 1)
        context = context.transpose(0, 1)

        e = self.attn(context)
        # e: B * Q * C
        e = torch.matmul(query, e.transpose(1, 2))
        if inf_mask is not None:
            e = e + inf_mask.unsqueeze(1)

        # dim w: B * Q * C, context: B * C * H, wanted B * Q * H
        w = F.softmax(e, dim=2)
        c = torch.matmul(w, context)
        # # Return the softmax normalized probability scores (with added dimension
        if requires_weight:
            return c.transpose(0, 1), w
        return c.transpose(0, 1)