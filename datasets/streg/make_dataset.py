from os.path import join
import sys
import numpy as np
import pickle
from grammar.grammar import Grammar

# from grammar.hypothesis import Hypothesis, ApplyRuleAction
# from components.action_info import get_action_infos
from components.dataset import Example
# from components.vocab import VocabEntry, Vocab

from grammar.streg.streg_transition_system import *
from datasets.utils import build_dataset_vocab

def _read_lines(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [x.rstrip('\n') for x in lines]

    return lines

class StReg:
    @staticmethod
    def load_map_file(filename):
        with open(filename) as f:
            lines = f.readlines()
        lines = [x.rstrip() for x in lines]
        maps = []
        for l in lines:
            fields = l.split(" ")
            num = int(fields[0])
            fields = fields[1:]
            if num == 0:
                maps.append([])
                continue
            m = []
            for f in fields:
                pair = f.split(",", 1)
                m.append((pair[0], pair[1]))
            maps.append(m)
        return maps

    @staticmethod
    def load_examples(filename):
        lines = _read_lines(filename)
        lines = [x.split(" ") for x in lines]
        lines = [[(y.split(",", 1)[0], y.split(",", 1)[1])
                    for y in x] for x in lines]
        return lines

    @staticmethod
    def load_rec(filename):
        with open(filename, "rb") as f:
            rec = pickle.load(f)
        return rec


def load_dataset(transition_system, split):
    prefix = 'data/streg/'
    src_file = join(prefix, "src-{}.txt".format(split))
    spec_file = join(prefix, "targ-{}.txt".format(split))
    map_file = join(prefix, "map-{}.txt".format(split))
    exs_file = join(prefix, "exs-{}.txt".format(split))
    rec_file = join(prefix, "rec-{}.pkl".format(split))

    exs_info = StReg.load_examples(exs_file)
    map_info = StReg.load_map_file(map_file)
    rec_info = StReg.load_rec(rec_file)

    examples = []
    for idx, (src_line, spec_line, str_exs, cmap, rec) in enumerate(zip(open(src_file), open(spec_file), exs_info, map_info, rec_info)):
        print(idx)
        
        src_line = src_line.rstrip()
        spec_line = spec_line.rstrip()
        src_toks = src_line.split()
        
        spec_toks = spec_line.rstrip().split()
        spec_ast = streg_expr_to_ast(transition_system.grammar, spec_toks)
        # sanity check
        reconstructed_expr = transition_system.ast_to_surface_code(spec_ast)
        # print("Spec", spec_line)
        # print("Rcon", reconstructed_expr)
        assert spec_line == reconstructed_expr
        tgt_action_tree = transition_system.get_action_tree(spec_ast)
        ast_from_action = transition_system.build_ast_from_actions(tgt_action_tree)
        assert is_equal_ast(ast_from_action, spec_ast)

        expr_from_hyp = transition_system.ast_to_surface_code(ast_from_action)
        assert expr_from_hyp == spec_line
        
        example = Example(idx=idx,
                        src_toks=src_toks,
                        tgt_actions=tgt_action_tree,
                        tgt_toks=spec_toks,
                        tgt_ast=spec_ast,
                        meta={"str_exs": str_exs,
                            "const_map": cmap,
                            "worker_info": rec})

        examples.append(example)
    return examples

def make_dataset():
    grammar = Grammar.from_text(open('data/streg/streg_asdl.txt').read())
    transition_system = StRegTransitionSystem(grammar)
    train_set = load_dataset(transition_system, "train")
    val_set = load_dataset(transition_system, "val")
    testi_set = load_dataset(transition_system, "testi")
    teste_set = load_dataset(transition_system, "teste")

    # generate vocabulary
    vocab_freq_cutoff = 2
    vocab = build_dataset_vocab(train_set, transition_system, src_cutoff=vocab_freq_cutoff)

    pickle.dump(train_set, open('data/streg/train.bin', 'wb'))
    pickle.dump(val_set, open('data/streg/dev.bin', 'wb'))
    pickle.dump(testi_set, open('data/streg/testi.bin', 'wb'))
    pickle.dump(teste_set, open('data/streg/teste.bin', 'wb'))
    pickle.dump(vocab, open('data/streg/vocab.bin', 'wb'))

if __name__ == "__main__":
    make_dataset()
