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
from models.ASN import EmbeddingLayer, RNNEncoder


class CompositeTypeDecoder(nn.Module):
    def __init__(self, args, type, productions):
        super().__init__()
        self.type = type
        self.productions = productions
        self.w = nn.Linear(2 * args.io_hid_size, len(productions))

    def forward(self, x):
        return self.w(x)

    def logits(self, x):
        return self.w(x)

    def probs(self, x):
        return torch.sigmoid(self.w(x))
        
class PrimitiveTypeDecoder(nn.Module):
    def __init__(self, args, type, vocab):
        super().__init__()
        self.type = type
        self.vocab = vocab
        self.w = nn.Linear(2 * args.io_hid_size, len(vocab))

    def forward(self, x):
        return self.w(x)
    
    def logits(self, x):
        return self.w(x)

    def probs(self, x):
        return torch.sigmoid(self.w(x))

class DeepCoder(nn.Module):
    def __init__(self, args, transition_system, vocab, io_vocab):
        super().__init__()
        self.args = args
        self.transition_system = transition_system
        self.vocab = vocab
        self.io_vocab = io_vocab
        grammar = transition_system.grammar
        self.grammar = grammar

        self.io_embedding = EmbeddingLayer(args.io_emb_size, io_vocab.size(), args.dropout)
        self.io_encoder = RNNEncoder(2 * args.io_emb_size, args.io_hid_size, args.dropout, True, reduce=False)
        self.dropout = nn.Dropout(args.dropout)
        # init
        comp_type_decoders = []
        for dsl_type in grammar.composite_types:
            comp_type_decoders.append((dsl_type.name,
                                      CompositeTypeDecoder(args, dsl_type, grammar.get_prods_by_type(dsl_type))))
        self.comp_type_dict = nn.ModuleDict(comp_type_decoders)
    
        prim_type_decoders = []
        for prim_type in grammar.primitive_types:
            prim_type_decoders.append((prim_type.name,
                                      PrimitiveTypeDecoder(args, prim_type, vocab.primitive_vocabs[prim_type])))
        self.prim_type_dict = nn.ModuleDict(prim_type_decoders)

        self.pooling = args.io_pooling

    def score(self, examples):
        # for ex in examples:
        scores = [self._score(ex) for ex in examples]
        # print(scores)
        return torch.stack(scores)

    def _score(self, ex):
        batch = CIOBatch([ex], self.grammar, self.vocab, self.io_vocab)

        # _, encoder_outputs = self.
        # max_pooling
        _, (enc_outputs, _) = self.encode_io(batch)
        if self.pooling == 'max':
            enc_outputs, _ = torch.max(enc_outputs, dim=0)
        elif self.pooling == 'mean':
            enc_outputs = torch.mean(enc_outputs, dim=0)

        enc_outputs = self.dropout(enc_outputs)
        score = 0.
        for dsl_type_name, x in self.comp_type_dict.items():
            logs = x.logits(enc_outputs)
            loss = F.binary_cross_entropy_with_logits(logs, batch.targets[dsl_type_name] , reduction='sum')
            score += loss

        for prim_type_name, x in self.prim_type_dict.items():
            logs = x.logits(enc_outputs)
            loss = F.binary_cross_entropy_with_logits(logs, batch.targets[prim_type_name] , reduction='sum')
            score += loss

        return score

    def predict_prior(self, ex):
        batch = CIOBatch([ex], self.grammar, self.vocab, self.io_vocab)

        _, (enc_outputs, _) = self.encode_io(batch)
        if self.pooling == 'max':
            enc_outputs, _ = torch.max(enc_outputs, dim=0)
        elif self.pooling == 'mean':
            enc_outputs = torch.mean(enc_outputs, dim=0)

        prior_dict = {}
        for dsl_type_name, x in self.comp_type_dict.items():
            logits = x.probs(enc_outputs)
            prior_dict[dsl_type_name] = logits

        for prim_type_name, x in self.prim_type_dict.items():
            logits = x.probs(enc_outputs)
            prior_dict[prim_type_name] = logits
    
        return prior_dict
    
    def predict_annotated_prior(self, ex):
        batch = CIOBatch([ex], self.grammar, self.vocab, self.io_vocab)

        _, (enc_outputs, _) = self.encode_io(batch)
        if self.pooling == 'max':
            enc_outputs, _ = torch.max(enc_outputs, dim=0)
        elif self.pooling == 'mean':
            enc_outputs = torch.mean(enc_outputs, dim=0)

        prior_dict = {}
        for dsl_type_name, x in self.comp_type_dict.items():
            logits = x.probs(enc_outputs).cpu()
            prior_dict[dsl_type_name] = 'rule', logits, x.productions

        for prim_type_name, x in self.prim_type_dict.items():
            logits = x.probs(enc_outputs).cpu()
            prior_dict[prim_type_name] = 'tok', logits, x.vocab
    
        return prior_dict

    def score_and_prior(self, ex):
        batch = CIOBatch([ex], self.grammar, self.vocab, self.io_vocab)

        # _, encoder_outputs = self.
        # max_pooling
        _, (enc_outputs, _) = self.encode_io(batch)
        if self.pooling == 'max':
            enc_outputs, _ = torch.max(enc_outputs, dim=0)
        elif self.pooling == 'mean':
            enc_outputs = torch.mean(enc_outputs, dim=0)

        enc_outputs = self.dropout(enc_outputs)
        score = 0.
        prior_dict = {}
        for dsl_type_name, x in self.comp_type_dict.items():
            logs = x.logits(enc_outputs)
            loss = F.binary_cross_entropy_with_logits(logs, batch.targets[dsl_type_name] , reduction='sum')
            prior_dict[dsl_type_name] = torch.sigmoid(logs)
            score += loss

        for prim_type_name, x in self.prim_type_dict.items():
            logs = x.logits(enc_outputs)
            loss = F.binary_cross_entropy_with_logits(logs, batch.targets[prim_type_name] , reduction='sum')
            prior_dict[prim_type_name] = torch.sigmoid(logs)
            score += loss
        return score, prior_dict

    def match_test(self, ex):
        batch = CIOBatch([ex], self.grammar, self.vocab, self.io_vocab)

        _, (enc_outputs, _) = self.encode_io(batch)
        if self.pooling == 'max':
            enc_outputs, _ = torch.max(enc_outputs, dim=0)
        elif self.pooling == 'mean':
            enc_outputs = torch.mean(enc_outputs, dim=0)

        flag = True
        num_acc = 0
        cnt = 0
        score = 0.
        l1_loss = 0.
        for dsl_type_name, x in self.comp_type_dict.items():
            logits = x.logits(enc_outputs)

            loss = F.binary_cross_entropy_with_logits(logits, batch.targets[dsl_type_name] , reduction='sum')
            score += loss

            probs = torch.sigmoid(logits).numpy()
            preds = probs > 0.5

            targets = batch.targets[dsl_type_name].numpy() > 0
            l1_loss += np.sum(np.abs(targets - probs))
            num_acc += np.sum(preds == targets)
            cnt += preds.size
            if not np.all(preds == targets):
                flag = False

        for prim_type_name, x in self.prim_type_dict.items():
            logits = x.logits(enc_outputs)

            loss = F.binary_cross_entropy_with_logits(logits, batch.targets[prim_type_name] , reduction='sum')
            score += loss

            probs = torch.sigmoid(logits).numpy()
            preds = probs > 0.5

            targets = batch.targets[prim_type_name].numpy() > 0
            l1_loss += np.sum(np.abs(targets - probs))
            num_acc += np.sum(preds == targets)
            cnt += preds.size
            if not np.all(preds == targets):
                flag = False

        return flag, num_acc / cnt, score.item(), l1_loss

    def encode_io(self, batch):

        io_lens = batch.io_lens
        # sent
        exs_embedding =  self.io_embedding(batch.exs_toks)
        cls_embedding = self.io_embedding(batch.cls_toks)
        io_embedding = torch.cat((exs_embedding, cls_embedding), dim=2)        
        context_vecs, final_state = self.io_encoder(io_embedding, io_lens)
        # L * b * hidden,  
        # print(context_vecs.size(), final_state[0].size(), final_state[1].size())
        return context_vecs, final_state

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
        transition_system = params['transition_system']
        saved_args = params['args']
        io_vocab = params['io_vocab']
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