import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from components.dataset import Batch, CIOBatch
from grammar.transition_system import ApplyRuleAction, GenTokenAction, PlaceHolderAction, ActionTree
from grammar.hypothesis import Hypothesis
import numpy as np
import os
from common.config import update_args

from grammar.streg.streg_transition_system import partial_asdl_ast_to_streg_ast, batch_preverify_regex_with_exs, asdl_ast_to_streg_ast, is_equal_ast, is_partial_ast
from models.ASN import CompositeTypeModule, ConstructorTypeModule, PrimitiveTypeModule, LuongAttention, RNNEncoder, EmbeddingLayer

class RobustFill(nn.Module):
    def __init__(self, args, transition_system, vocab, io_vocab):
        super().__init__()

        self.args = args

        # encoder part
        self.io_embedding = EmbeddingLayer(args.io_emb_size, io_vocab.size(), args.dropout)
        self.io_encoder = RNNEncoder(2 * args.io_emb_size, args.io_hid_size, args.dropout, True, reduce=True)

        # decoder part
        self.transition_system = transition_system
        self.vocab = vocab
        self.io_vocab = io_vocab
        grammar = transition_system.grammar
        self.grammar = grammar
        # init
        module_input_size = 2 * args.io_hid_size + args.enc_hid_size
        comp_type_modules = []
        for dsl_type in grammar.composite_types:
            comp_type_modules.append((dsl_type.name,
                                      CompositeTypeModule(args, dsl_type, grammar.get_prods_by_type(dsl_type), module_input_size)))
        self.comp_type_dict = nn.ModuleDict(comp_type_modules)

        # init
        cnstr_type_modules = []
        for prod in grammar.productions:
            cnstr_type_modules.append((prod.constructor.name,
                                       ConstructorTypeModule(args, prod, module_input_size)))
        self.const_type_dict = nn.ModuleDict(cnstr_type_modules)

        prim_type_modules = []
        for prim_type in grammar.primitive_types:
            prim_type_modules.append((prim_type.name,
                                      PrimitiveTypeModule(args, prim_type, vocab.primitive_vocabs[prim_type],module_input_size)))
        self.prim_type_dict = nn.ModuleDict(prim_type_modules)

        self.v_lstm = nn.LSTM(args.enc_hid_size, args.enc_hid_size)
        self.io_attn = LuongAttention(args.enc_hid_size, 2 * args.io_hid_size)
        self.dropout = nn.Dropout(args.dropout)

    def score(self, examples):
        # for ex in examples:
        scores = [self._score(ex) for ex in examples]
        return torch.stack(scores)

    def _score(self, ex):
        batch = Batch([ex], self.grammar, self.vocab)

        io_batch = CIOBatch([ex], self.grammar, self.vocab, self.io_vocab, train=False)
        io_context_vecs, (enc_outputs, enc_cells) = self.encode_io(io_batch)
        enc_outputs, _ = torch.max(enc_outputs, dim=0, keepdim=True)
        enc_cells, _ = torch.max(enc_cells, dim=0, keepdim=True)
        init_state = (enc_outputs, enc_cells)

        return self._score_node(self.grammar.root_type, init_state, ex.tgt_actions, io_context_vecs, io_batch.io_masks)

    def encode_io(self, batch):

        io_lens = batch.io_lens
        # sent
        exs_embedding =  self.io_embedding(batch.exs_toks)
        cls_embedding = self.io_embedding(batch.cls_toks)
        io_embedding = torch.cat((exs_embedding, cls_embedding), dim=2)
        context_vecs, final_state = self.io_encoder(io_embedding, io_lens)
        # L * b * hidden,  
        return context_vecs, final_state

    def _score_node(self, node_type, v_state, action_node, io_context_vecs, io_masks):
        v_output = self.dropout(v_state[0])
        io_contexts = self.io_attn(v_output.unsqueeze(0), io_context_vecs).squeeze(0)
        io_contexts, _ = torch.max(io_contexts, dim=0, keepdim=True)
        if node_type.is_primitive_type():
            module = self.prim_type_dict[node_type.name]
            # scores = mask * module()
            scores = module.score(v_output, io_contexts)
            # scores =  [tgt_action_tree.action.choice_idx]
            # b * choice
            score = -1 * scores.view([-1])[action_node.action.choice_index]
            return score

        
        cnstr = action_node.action.choice.constructor
        comp_module = self.comp_type_dict[node_type.name]
        scores = comp_module.score(v_output, io_contexts)
        score = -1 * scores.view([-1])[action_node.action.choice_index]

        # pass through
        cnstr_module = self.const_type_dict[cnstr.name]
        # cnstr_results = const_module.iup()
        # next_states = self.v_lstm( [1 * 1 * x], v_state)
        cnstr_results = cnstr_module.update(self.v_lstm, v_state, io_contexts)
        for next_field, next_state, next_action in zip(cnstr.fields, cnstr_results, action_node.fields):
            score += self._score_node(next_field.type, next_state, next_action, io_context_vecs, io_masks)
        return score

    def naive_parse(self, ex):
        batch = Batch([ex], self.grammar, self.vocab, train=False)        
        
        io_batch = CIOBatch([ex], self.grammar, self.vocab, self.io_vocab, train=False)
        io_context_vecs, (enc_outputs, enc_cells) = self.encode_io(io_batch)
        enc_outputs, _ = torch.max(enc_outputs, dim=0, keepdim=True)
        enc_cells, _ = torch.max(enc_cells, dim=0, keepdim=True)
        init_state = (enc_outputs, enc_cells)

        action_tree = self._naive_parse(self.grammar.root_type, init_state, io_context_vecs, io_batch.io_masks, 1)

        return self.transition_system.build_ast_from_actions(action_tree)

    def _naive_parse(self, node_type, v_state, io_context_vecs, io_masks, depth):

        # v_state = v_state.torch.unsqueeze(0)

        # tgt_production if production needed
        # tgt_production = tgt

        # else token needed
        # tgt_token = tgt
        io_contexts = self.io_attn(v_state[0].unsqueeze(0), io_context_vecs).squeeze(0)
        io_contexts, _ = torch.max(io_contexts, dim=0, keepdim=True)
        if depth > 9:
            return ActionTree(PlaceHolderAction(node_type))

        if node_type.is_primitive_type():
            module = self.prim_type_dict[node_type.name]
            # scores = mask * module()
            scores = module.score(v_state[0], io_contexts).cpu().numpy().flatten()
            # scores =  [tgt_action_tree.action.choice_idx]
            # b * choice
            # score = -1 * scores.view([-1])[action_node.action.choice_index]
            choice_idx = np.argmax(scores)
            return ActionTree(GenTokenAction(node_type, module.vocab.get_word(choice_idx)))

        
        comp_module = self.comp_type_dict[node_type.name]
        scores = comp_module.score(v_state[0], io_contexts).cpu().numpy().flatten()
        choice_idx = np.argmax(scores)
        production = comp_module.productions[choice_idx]

        action = ApplyRuleAction(node_type, production)
        cnstr = production.constructor

        # pass through
        cnstr_module = self.const_type_dict[cnstr.name]
        # cnstr_results = const_module.iup()
        # next_states = self.v_lstm( [1 * 1 * x], v_state)
        cnstr_results = cnstr_module.update(self.v_lstm, v_state, io_contexts)
        action_fields = [self._naive_parse(next_field.type, next_state, io_context_vecs, io_masks, depth+1) for next_field, next_state in zip(cnstr.fields, cnstr_results)]

        return ActionTree(action, action_fields)


    def ex_guided_parse(self, ex, cache=None):
        return self.ex_guided_parse_and_track_budget(ex, cache=cache)
    
    def precompute_scores_for_holes(self, hyp, io_context_vecs, io_masks):
        # v_state = v_state.torch.unsqueeze(0)

        # tgt_production if production needed
        # tgt_production = tgt

        # else token needed
        # tgt_token = tgt
        
        nodes_to_proc = hyp.get_unprocessed_nodes()

        for pending_node in nodes_to_proc:
            v_state = pending_node.action.v_state

            io_contexts = self.io_attn(v_state[0].unsqueeze(0), io_context_vecs).squeeze(0)
            io_contexts, _ = torch.max(io_contexts, dim=0, keepdim=True)
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
                scores = module.score(v_state[0], io_contexts)
                scores = scores.squeeze(0)
                pending_node.action.cache_scores(scores, module.vocab, smooth_factor=smooth_factor)
            else:
                comp_module = self.comp_type_dict[node_type.name]
                scores = comp_module.score(v_state[0], io_contexts)
                scores = scores.squeeze(0)
                pending_node.action.cache_scores(scores, comp_module.productions, smooth_factor=smooth_factor)

    
        hyp.update_priority_scores()

    def ex_guided_parse_and_track_budget(self, ex, cache=None):
        batch = Batch([ex], self.grammar, self.vocab)

        io_batch = CIOBatch([ex], self.grammar, self.vocab, self.io_vocab, train=False)
        io_context_vecs, (enc_outputs, enc_cells) = self.encode_io(io_batch)
        enc_outputs, _ = torch.max(enc_outputs, dim=0, keepdim=True)
        enc_cells, _ = torch.max(enc_cells, dim=0, keepdim=True)
        init_state = (enc_outputs, enc_cells)
        
        completed_hyps = []
        # cur_beam = [Hypothesis.init_hypothesis(self.grammar.root_type, init_state)]
        init_hyp = Hypothesis.init_hypothesis(self.grammar.root_type, init_state, order=self.args.search_order)
        self.precompute_scores_for_holes(init_hyp, io_context_vecs, io_batch.io_masks)
        cur_beam = [init_hyp]
        num_executed = 0

        for ts in range(self.args.max_decode_step):
            hyp_pools = []
            for hyp in cur_beam:
                continuations = self.continuations_of_hyp_with_cache(hyp, io_context_vecs, io_batch.io_masks)
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
                    self.precompute_scores_for_holes(hyp, io_context_vecs, io_batch.io_masks)
                    cur_beam.append(hyp)
            if len(hyp_pools) < num_slots:
                num_executed = cur_executed + len(all_scores_this_time)
            if not cur_beam:
                break
        print('Final Num', num_executed)
        completed_hyps.sort(key=lambda x: x.score, reverse=True)
        return completed_hyps

    def parse(self, ex):    
        batch = Batch([ex], self.grammar, self.vocab)

        io_batch = CIOBatch([ex], self.grammar, self.vocab, self.io_vocab, train=False)
        io_context_vecs, (enc_outputs, enc_cells) = self.encode_io(io_batch)
        enc_outputs, _ = torch.max(enc_outputs, dim=0, keepdim=True)
        enc_cells, _ = torch.max(enc_cells, dim=0, keepdim=True)
        init_state = (enc_outputs, enc_cells)

        completed_hyps = []
        init_hyp = Hypothesis.init_hypothesis(self.grammar.root_type, init_state, order=self.args.search_order)
        self.precompute_scores_for_holes(init_hyp, io_context_vecs, io_batch.io_masks)
        cur_beam = [init_hyp]
        for ts in range(self.args.max_decode_step):
            hyp_pools = []
            for hyp in cur_beam:
                continuations = self.continuations_of_hyp_with_cache(hyp, io_context_vecs, io_batch.io_masks)
                hyp_pools.extend(continuations)
            
            hyp_pools.sort(key=lambda x: x.score, reverse=True)
            # next_beam = next_beam[:self.args.beam_size]
            
            num_slots = self.args.beam_size - len(completed_hyps)

            cur_beam = []
            for hyp_i, hyp  in enumerate(hyp_pools[:num_slots]):
                if hyp.is_complete():
                    completed_hyps.append(hyp)
                else:
                    self.precompute_scores_for_holes(hyp, io_context_vecs, io_batch.io_masks)
                    cur_beam.append(hyp)
            
            if not cur_beam:
                break

        completed_hyps.sort(key=lambda x: x.score, reverse=True)
        return completed_hyps

    def continuations_of_hyp_with_cache(self, hyp, io_context_vecs, io_masks):        
        pending_node = hyp.get_pending_node()
        v_state = pending_node.action.v_state
        
    
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

        io_contexts = self.io_attn(v_state[0].unsqueeze(0), io_context_vecs).squeeze(0)
        io_contexts, _ = torch.max(io_contexts, dim=0, keepdim=True)


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
            cnstr_results = cnstr_module.update(self.v_lstm, v_state, io_contexts)
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
            'io_vocab': self.io_vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, filename)

    @classmethod
    def load(cls, model_path, ex_args=None, cuda=False):
        params = torch.load(model_path)
        vocab = params['vocab']
        io_vocab = params['io_vocab']
        transition_system = params['transition_system']
        saved_args = params['args']
        # update saved args
        saved_state = params['state_dict']
        saved_args.cuda = cuda
        if ex_args:
            update_args(saved_args, ex_args)
        parser = cls(saved_args, transition_system, vocab, io_vocab)
        parser.load_state_dict(saved_state)
        
        # setattr(saved_args, )
        if cuda: parser = parser.cuda()
        parser.eval()

        return parser
