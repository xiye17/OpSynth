# coding=utf-8
from collections import OrderedDict

import torch
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle


class Dataset:
    def __init__(self, examples):
        self.examples = examples

    @property
    def all_source(self):
        return [e.src_sent for e in self.examples]

    @property
    def all_targets(self):
        return [e.tgt_code for e in self.examples]

    @staticmethod
    def from_bin_file(file_path):
        examples = pickle.load(open(file_path, 'rb'))
        return Dataset(examples)

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def batch_iter(self, batch_size, shuffle=False):
        index_arr = np.arange(len(self.examples))
        if shuffle:
            np.random.shuffle(index_arr)

        batch_num = int(np.ceil(len(self.examples) / float(batch_size)))
        for batch_id in range(batch_num):
            batch_ids = index_arr[batch_size *
                                  batch_id: batch_size * (batch_id + 1)]
            batch_examples = [self.examples[i] for i in batch_ids]
            batch_examples.sort(key=lambda e: -len(e.src_toks))

            yield batch_examples


class Example:
    def __init__(self, src_toks, tgt_toks, tgt_ast, idx=0, tgt_actions=None, meta=None):
        self.src_toks = src_toks
        self.tgt_toks = tgt_toks
        self.tgt_ast = tgt_ast
        self.tgt_actions = tgt_actions

        self.idx = idx
        self.meta = meta

    def tokenize_exs(self):
        exs = self.meta['str_exs'] 
        exs_toks = []
        const_map = self.meta['const_map']
        for sig, ex_str in exs:
            toked_ex = []
            if sig == '+':
                toked_ex.append('<pos>')
            else:
                toked_ex.append('<neg>')
            toked_ex.extend(_tokenize_exs_with_map(ex_str, const_map))
            exs_toks.append(toked_ex)
        cls_toks = [
            [_class_for_exs(t) for t in ex_t]
            for ex_t in exs_toks
        ]
        self.exs_toks = exs_toks
        self.cls_toks = cls_toks

def _class_for_exs(t):
    if t  in ['<pos>', '<neg>']:
        return '<label>'
    if len(t) == 1:
        if 'a' <= t and t <= 'z':
            return '<low>'
        elif 'A' <= t and t <= 'Z':
            return '<cap>'
        elif '0' <= t and t <= '9':
            return '<num>'
        else:
            return '<spec>'
    else:
        if t.startswith('const'):
            return '<cnst>'
        else:
            raise ValueError(t)
        
        
def _tokenize_exs_with_map(ex, const_m):
    # l_ex = len(ex)
    left = ex
    toks = []
    while left:
        flag = False
        for c, s in const_m:
            if left.startswith(s):
                flag = True
                toks.append(c)
                left = left[len(s):]
                break
        if flag:
            continue
        toks.append(left[0])
        left = left[1:]
    return toks
    # exit()
def sent_lens_to_mask(lens, max_length):
    mask = [[1 if j < l else 0 for j in range(max_length)] for l in lens]
    # match device of input
    return mask

class Batch(object):
    def __init__(self, examples, grammar, vocab, train=True, cuda=False):
        self.examples = examples

        # self.src_sents = [e.src_sent for e in self.examples]
        # self.src_sents_len = [len(e.src_sent) for e in self.examples]

        self.grammar = grammar
        self.vocab = vocab
        self.cuda = cuda
        self.train = train
        self.build_input()

    def __len__(self):
        return len(self.examples)

    def build_input(self):

        sent_lens = [len(x.src_toks) for x in self.examples]
        max_sent_len = max(sent_lens)
        sent_masks = sent_lens_to_mask(sent_lens, max_sent_len)
        sents = [
            [
                self.vocab.src_vocab[e.src_toks[i]] if i < l else self.vocab.src_vocab['<pad>']
                for i in range(max_sent_len)
            ]
            for l, e in zip(sent_lens, self.examples)
        ]
        self.sents = torch.LongTensor(sents)
        self.sent_lens = torch.LongTensor(sent_lens)
        self.sent_masks = torch.ByteTensor(sent_masks)
        if self.train:
            [self.compute_choice_index(e.tgt_actions) for e in self.examples]

    def compute_choice_index(self, node):
        if node.action.action_type == "ApplyRule":
            candidate = self.grammar.get_prods_by_type(node.action.type)
            node.action.choice_index = candidate.index(node.action.choice)
            [self.compute_choice_index(x) for x in node.fields]
        elif node.action.action_type == "GenToken":
            token_vocab = self.vocab.primitive_vocabs[node.action.type]
            node.action.choice_index = token_vocab[node.action.choice]
        else:
            raise ValueError("invalid action type", node.action.action_type)


# def _tokens
def _index_io_pair(io_pair, io_vocab):
    head = io_vocab['<pos>'] if io_pair[0] == '+' else io_vocab['<neg>']
    chars = [io_vocab[x] for x in io_pair[1]]
    
    return [head] + chars


# we call it batch, but it only supports single example for now
# class IOBatch:
#     # list of 
#     def __init__(self, examples, grammar, vocab, io_vocab, train=True, cuda=False):
#         self.examples = examples
#         self.grammar = grammar
#         self.io_vocab = io_vocab
#         self.vocab = vocab
#         self.cuda = cuda
#         self.train = train
#         self.build_input()
    
        
#     def build_input(self):
#         io_pairs = [x.meta['str_exs'] for x in self.examples]
#         io_pairs = io_pairs[0]
#         io_pairs.sort(key = lambda x: len(x[1]), reverse=True)
#         io_pairs = [_index_io_pair(x, self.io_vocab) for x in io_pairs]
#         io_lens = [len(x) for x in io_pairs]
#         max_io_len = max(io_lens)
#         io_masks = sent_lens_to_mask(io_lens, max_io_len)

#         ios = [
#             [
#                 e[i] if i < l else self.io_vocab['<pad>']
#                 for i in range(max_io_len)
#             ]
#             for l, e in zip(io_lens, io_pairs)
#         ]

#         self.ios = torch.LongTensor(ios)
#         self.io_lens = torch.LongTensor(io_lens)
#         self.io_masks = torch.ByteTensor(io_masks)

#         if self.train:
#             targets = {}
#             # init targets
#             for dsl_type in self.grammar.composite_types:
#                 targets[dsl_type.name] = [0] * len(self.grammar.get_prods_by_type(dsl_type))
#             for prim_type in self.grammar.primitive_types:
#                 targets[prim_type.name] = [0] * self.vocab.primitive_vocabs[prim_type].size()
#             [self.compute_choice_index(targets, e.tgt_actions) for e in self.examples]            
#             for t in targets:
#                 targets[t] = torch.Tensor(targets[t])

#             self.targets = targets

#     def compute_choice_index(self, targets, node):
#         if node.action.action_type == "ApplyRule":
#             candidate = self.grammar.get_prods_by_type(node.action.type)
#             targets[node.action.type.name][candidate.index(node.action.choice)] = 1
#             [self.compute_choice_index(targets, x) for x in node.fields]
#         elif node.action.action_type == "GenToken":
#             token_vocab = self.vocab.primitive_vocabs[node.action.type]
#             targets[node.action.type.name][token_vocab[node.action.choice]] = 1
#         else:
#             raise ValueError("invalid action type", node.action.action_type)



# we call it a cannonicalized batch, but it only supports single example for now
class CIOBatch:
    # list of 
    def __init__(self, examples, grammar, vocab, io_vocab, train=True, cuda=False):
        self.examples = examples
        self.grammar = grammar
        self.io_vocab = io_vocab
        self.vocab = vocab
        self.cuda = cuda
        self.train = train
        self.build_input()
        
    def build_input(self):
        exs_toks = self.examples[0].exs_toks
        cls_toks = self.examples[0].cls_toks

        exs_toks.sort(key = lambda x: len(x), reverse=True)
        cls_toks.sort(key = lambda x: len(x), reverse=True)
        io_lens = [len(x) for x in exs_toks]
        max_io_len = max(io_lens)
        io_masks = sent_lens_to_mask(io_lens, max_io_len)
        exs_toks = [
            [
                self.io_vocab[e[i]] if i < l else self.io_vocab['<pad>']
                for i in range(max_io_len)
            ]
            for l, e in zip(io_lens, exs_toks)
        ]
        cls_toks = [
            [
                self.io_vocab[e[i]] if i < l else self.io_vocab['<pad>']
                for i in range(max_io_len)
            ]
            for l, e in zip(io_lens, cls_toks)
        ]
        self.exs_toks = torch.LongTensor(exs_toks)
        self.cls_toks = torch.LongTensor(cls_toks)
        self.io_lens = torch.LongTensor(io_lens)
        self.io_masks = torch.ByteTensor(io_masks)

        if self.train:
            targets = {}
            # init targets
            for dsl_type in self.grammar.composite_types:
                targets[dsl_type.name] = [0] * len(self.grammar.get_prods_by_type(dsl_type))
            for prim_type in self.grammar.primitive_types:
                targets[prim_type.name] = [0] * self.vocab.primitive_vocabs[prim_type].size()
            [self.compute_choice_index(targets, e.tgt_actions) for e in self.examples]            
            for t in targets:
                targets[t] = torch.Tensor(targets[t])

            self.targets = targets

    def compute_choice_index(self, targets, node):
        if node.action.action_type == "ApplyRule":
            candidate = self.grammar.get_prods_by_type(node.action.type)
            targets[node.action.type.name][candidate.index(node.action.choice)] = 1
            [self.compute_choice_index(targets, x) for x in node.fields]
        elif node.action.action_type == "GenToken":
            token_vocab = self.vocab.primitive_vocabs[node.action.type]
            targets[node.action.type.name][token_vocab[node.action.choice]] = 1
        else:
            raise ValueError("invalid action type", node.action.action_type)


class SynthAction:
    # label correct wrong irrelevent
    def __init__(self, v_state, action, label, score, hole=None, node=None):
        self.v_state = v_state
        # self.hole = hole
        self.action = action
        self.label = label
        self.score = score
        self.hole = hole
        self.node = node

class SynthBranch:
    # things needed
    # sequence of actions
    # hidden states of the action based on 
    # action needs a label, irrelevant, ground_truth, diverged
    def __init__(self, actions, hypothesis, parent_ptrs=None):
        self.actions = actions
        self.hypothesis = hypothesis
        self.parent_ptrs = parent_ptrs

class SynthPath:
    def __init__(self, branches):
        self.branches = branches
