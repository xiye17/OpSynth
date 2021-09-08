from common.config import *
from components.dataset import *

from grammar.grammar import Grammar

from grammar.streg.streg_transition_system import StRegTransitionSystem
from models.ASN import ASNParser
from models import nn_utils

from torch import optim
import os
import time

def train(args):
    train_set = Dataset.from_bin_file(args.train_file)
    if args.dev_file:
        dev_set = Dataset.from_bin_file(args.dev_file)
    else: dev_set = Dataset(examples=[])
    
    vocab = pickle.load(open(args.vocab, 'rb'))
    grammar = Grammar.from_text(open(args.asdl_file).read())
    # transition_system = Registrable.by_name(args.transition_system)(grammar)
    transition_system = StRegTransitionSystem(grammar)
    
    parser = ASNParser(args, transition_system, vocab)    
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


#     evaluator = Registrable.by_name(args.evaluator)(transition_system, args=args)
#     if args.cuda: model.cuda()
#     optimizer_cls = eval('torch.optim.%s' % args.optimizer)  # FIXME: this is evil!
#     optimizer = optimizer_cls(model.parameters(), lr=args.lr)

#     if args.uniform_init:
#         print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init), file=sys.stderr)
#         nn_utils.uniform_init(-args.uniform_init, args.uniform_init, model.parameters())
#     elif args.glorot_init:
#         print('use glorot initialization', file=sys.stderr)
#         nn_utils.glorot_init(model.parameters())

#     # load pre-trained word embedding (optional)
#     # if args.glove_embed_path:
#     #     print('load glove embedding from: %s' % args.glove_embed_path, file=sys.stderr)
#     #     glove_embedding = GloveHelper(args.glove_embed_path)
#     #     glove_embedding.load_to(model.src_embed, vocab.source)


#             # print(loss.data)
#             loss_val = torch.sum(loss).data.item()
#             report_loss += loss_val
#             report_examples += len(batch_examples)
#             loss = torch.mean(loss)

#             if args.sup_attention:
#                 att_probs = ret_val[1]
#                 if att_probs:
#                     sup_att_loss = -torch.log(torch.cat(att_probs)).mean()
#                     sup_att_loss_val = sup_att_loss.data[0]
#                     report_sup_att_loss += sup_att_loss_val

#                     loss += sup_att_loss

#             loss.backward()

#             # clip gradient
#             if args.clip_grad > 0.:
#                 grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

#             optimizer.step()

#             if train_iter % args.log_every == 0:
#                 log_str = '[Iter %d] encoder loss=%.5f' % (train_iter, report_loss / report_examples)
#                 if args.sup_attention:
#                     log_str += ' supervised attention loss=%.5f' % (report_sup_att_loss / report_examples)
#                     report_sup_att_loss = 0.

#                 print(log_str, file=sys.stderr)
#                 report_loss = report_examples = 0.

#         print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

#         if args.save_all_models:
#             model_file = args.save_to + '.iter%d.bin' % train_iter
#             print('save model to [%s]' % model_file, file=sys.stderr)
#             model.save(model_file)

#         # perform validation
#         if args.dev_file:
#             if epoch % args.valid_every_epoch == 0:
#                 print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
#                 eval_start = time.time()
#                 eval_results = evaluation.evaluate(dev_set.examples, model, evaluator, args,
#                                                    verbose=True, eval_top_pred_only=args.eval_top_pred_only)
#                 dev_score = eval_results[evaluator.default_metric]

#                 print('[Epoch %d] evaluate details: %s, dev %s: %.5f (took %ds)' % (
#                                     epoch, eval_results,
#                                     evaluator.default_metric,
#                                     dev_score,
#                                     time.time() - eval_start), file=sys.stderr)

#                 is_better = history_dev_scores == [] or dev_score > max(history_dev_scores)
#                 history_dev_scores.append(dev_score)
#         else:
#             is_better = True

#         if args.decay_lr_every_epoch and epoch > args.lr_decay_after_epoch:
#             lr = optimizer.param_groups[0]['lr'] * args.lr_decay
#             print('decay learning rate to %f' % lr, file=sys.stderr)

#             # set new lr
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr

#         if is_better:
#             patience = 0
#             model_file = args.save_to + '.bin'
#             print('save the current model ..', file=sys.stderr)
#             print('save model to [%s]' % model_file, file=sys.stderr)
#             model.save(model_file)
#             # also save the optimizers' state
#             torch.save(optimizer.state_dict(), args.save_to + '.optim.bin')
#         elif patience < args.patience and epoch >= args.lr_decay_after_epoch:
#             patience += 1
#             print('hit patience %d' % patience, file=sys.stderr)

#         if epoch == args.max_epoch:
#             print('reached max epoch, stop!', file=sys.stderr)
#             exit(0)

#         if patience >= args.patience and epoch >= args.lr_decay_after_epoch:
#             num_trial += 1
#             print('hit #%d trial' % num_trial, file=sys.stderr)
#             if num_trial == args.max_num_trial:
#                 print('early stop!', file=sys.stderr)
#                 exit(0)

#             # decay lr, and restore from previously best checkpoint
#             lr = optimizer.param_groups[0]['lr'] * args.lr_decay
#             print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

#             # load model
#             params = torch.load(args.save_to + '.bin', map_location=lambda storage, loc: storage)
#             model.load_state_dict(params['state_dict'])
#             if args.cuda: model = model.cuda()

#             # load optimizers
#             if args.reset_optimizer:
#                 print('reset optimizer', file=sys.stderr)
#                 optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#             else:
#                 print('restore parameters of the optimizers', file=sys.stderr)
#                 optimizer.load_state_dict(torch.load(args.save_to + '.optim.bin'))

#             # set new lr
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr

#             # reset patience
#             patience = 0


if __name__ == '__main__':
    args = parse_args('train')
    train(args)
