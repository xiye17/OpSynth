from common.config import *
from components.dataset import *

from grammar.grammar import Grammar

from grammar.streg.streg_transition_system import StRegTransitionSystem
from models import nn_utils
from models.RobustFill import RobustFill

from torch import optim
import os
import time

def train(args):
    train_set = Dataset.from_bin_file(args.train_file)
    [x.tokenize_exs() for x in train_set]
    if args.dev_file:
        dev_set = Dataset.from_bin_file(args.dev_file)
        [x.tokenize_exs() for x in dev_set]
    else: dev_set = Dataset(examples=[])

    vocab = pickle.load(open(args.vocab, 'rb'))
    io_vocab = pickle.load(open(args.io_vocab, 'rb'))
    grammar = Grammar.from_text(open(args.asdl_file).read())
    # transition_system = Registrable.by_name(args.transition_system)(grammar)
    transition_system = StRegTransitionSystem(grammar)
    
    parser = RobustFill(args, transition_system, vocab, io_vocab)
    nn_utils.glorot_init(parser.parameters())

    optimizer = optim.Adam(parser.parameters(), lr=args.lr)
    best_acc = 0.0
    log_every = args.log_every
    
    train_begin = time.time()
    for epoch in range(1, args.max_epoch + 1):
        train_iter = 0
        loss_val = 0.
        epoch_loss = 0.

        parser.train()

        epoch_begin = time.time()
        for batch_example in train_set.batch_iter(batch_size=args.batch_size, shuffle=False):
            optimizer.zero_grad()
            loss = parser.score(batch_example)
            loss_val += torch.sum(loss).data.item()
            epoch_loss += torch.sum(loss).data.item()
            loss = torch.mean(loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parser.parameters(), args.clip_grad)

            optimizer.step()
            train_iter += 1
            if train_iter % log_every == 0:
                print("{:.3f}".format(loss_val / (log_every * args.batch_size )))
                loss_val = 0.

        # print(epoch, 'Train loss', '{:.3f}'.format(epoch_loss / len(train_set)), 'time elapsed %d' % (time.time() - epoch_begin))
        print('[epoch {}] train loss {:.3f}, epoch time {:.0f}, total time {:.0f}'.format(epoch, epoch_loss / len(train_set), time.time() - epoch_begin, time.time() - train_begin) )
        if epoch > args.run_val_after:
            eval_begin = time.time()
            parser.eval()
            with torch.no_grad():
                parse_results = [parser.naive_parse(ex) for ex in dev_set]
            match_results = [transition_system.compare_ast(e.tgt_ast, r) for e, r in zip(dev_set, parse_results)]
            match_acc = sum(match_results) * 1. / len(match_results)
            # print('Eval Acc', match_acc)
            print('[epoch {}] eval acc {:.3f}, eval time {:.0f}'.format(epoch, match_acc, time.time() - eval_begin))
            
            if match_acc >= best_acc:
                best_acc = match_acc
                parser.save(args.save_to)

            if args.save_all:
                model_file = args.save_to + '.iter%d.bin' % epoch
                print('save model to [%s]' % model_file)
                parser.save(model_file)



if __name__ == '__main__':
    args = parse_args('train_fill')
    train(args)
