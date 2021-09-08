import pickle
from os.path import join

from synthesizer.synthesizer import SearchTreeNode
from components.dataset import *
from grammar.transition_system import PlaceHolderAction
from components.vocab import VocabEntry
import math


def make_exs_vocab():
    train_set = pickle.load(open('data/streg/train.bin', 'rb'))

    ex_vocab = VocabEntry()
    ex_vocab.add('<pos>')
    ex_vocab.add('<neg>')
    
    for x in ['<label>', '<num>', '<low>', '<cap>', '<spec>', '<cnst>']:
        ex_vocab.add(x)
    for x in ['const0', 'const1', 'const2', 'const3', 'const4', 'const5', 'const6']:
        ex_vocab.add(x)
    all_toks = []
    for ex in train_set:
        # print(ex.meta['str_exs'])
        # for c in :
        # print(ex.meta['str_exs'][1])
        for str_e in ex.meta['str_exs']:
            all_toks.extend(str_e[1])
    all_toks = list(set(all_toks))
    all_toks.sort()
    for c in all_toks:
        ex_vocab.add(c)
    print(ex_vocab)
    print(list(ex_vocab.word_to_id.keys()))
    pickle.dump(ex_vocab , open('data/streg/io_vocab.bin', 'wb'))


def verify_exs(split):
    train_set = pickle.load(open('data/streg/%s.bin' % split, 'rb'))

    lens_ex = [len(x.meta['str_exs']) for x in train_set]


    print(set(lens_ex))

def main():
    make_exs_vocab()
    # verify_exs('train')

if __name__ == '__main__':
    main()