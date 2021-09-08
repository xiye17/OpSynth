from common.config import *
from components.dataset import *
from components.cache import *
from components.result import *
from grammar.grammar import Grammar

# from grammar.streg.streg_transition_system import 
from models.ASN import ASNParser
from models.RobASN import RobASN
from models import nn_utils

# from train_naive_searcher import FeatureVocab, IndexedFeature
from models.MetaLocator import RandomForestLocator
from synthesizer.synthesizer import EnumSynthesizer, FeedbackSynthesizer, LearnableSynthesizer, NoPruneSynthesizer
from synthesizer.graph_viz import make_dot_file
from torch import optim
import os
from os.path import join
import random
from common.utils import *
import subprocess
from tqdm import tqdm
import sys
from grammar.streg.streg_transition_system import init_backend, exit_backend

def post_process(x):
    x = x.replace("<m0>", "<!>")
    x = x.replace("<m1>", "<@>")
    x = x.replace("<m2>", "<#>")
    x = x.replace("<m3>", "<$>")
    x = x.replace(" ", "")
    return x

def check_equiv(spec0, spec1):
    if spec0 == spec1:
        # print("exact", spec0, spec1)
        return True
    # try:
    out = subprocess.check_output(
        ['java', '-cp', './external/datagen.jar:./external/lib/*', '-ea', 'datagen.Main', 'equiv',
            spec0, spec1], stderr=subprocess.DEVNULL)
    out = out.decode("utf-8")
    out = out.rstrip()
    # if out == "true":
    #     print("true", spec0, spec1)

    return out == "true"


def external_evalute_single(gt_spec, preds, exs, flag_force=False):
    pred_line = " ".join(preds)
    exs_line = " ".join(["{},{}".format(x[0], x[1]) for x in exs])
    flag_str = "true" if flag_force else "false"

    flag_use_file = len(preds) > 100
    if flag_use_file:
        filename = join("./external/", "eval_single_{}.in".format(random.random()))
        with open(filename, "w") as f:
            f.write(pred_line + "\n")
            f.write(exs_line + "\n")
            f.write(gt_spec)
        out = subprocess.check_output(
            ['java', '-cp', './external/datagen.jar:./external/lib/*', '-ea', 'datagen.Main', 'evaluate_single_file',
                filename, flag_str], stderr=subprocess.DEVNULL) 
        os.remove(filename)
    else:
        out = subprocess.check_output(
            ['java', '-cp', './external/datagen.jar:./external/lib/*', '-ea', 'datagen.Main', 'evaluate_single',
                pred_line, exs_line, gt_spec, flag_str], stderr=subprocess.DEVNULL)

    
    out = out.decode("utf-8")
    out = out.rstrip()
    vals = out.split(" ")
    return vals[0], vals[1:]

def inverse_regex_with_map(r, maps):
    for m in maps:
        src = m[0]
        if len(m[1]) == 1:
            dst = "<{}>".format(m[1])
        else:
            dst = "const(<{}>)".format(m[1])
        r = r.replace(src, dst)
    return r


def batch_filtering_test(gt, preds, meta, flag_force=False):
    gt = inverse_regex_with_map(gt, meta["const_map"])
    preds = [inverse_regex_with_map(x, meta["const_map"]) for x in preds]

    global_res, pred_res = external_evalute_single(gt, preds, meta["str_exs"], flag_force)
    if global_res in ["exact", "equiv"]:
        return True, global_res, pred_res
    else:
        return False, global_res, pred_res

def synthesize(args):
    test_set = Dataset.from_bin_file(args.test_file)
    # [x.tokenize_exs() for x in test_set]
    test_set.examples = test_set.examples[:10]
    # poi = [36, 38, 40, 64, 69, 70, 82, 84, 89]
    # test_set.examples = [test_set.examples[i] for i in poi]
    parser = ASNParser.load(args.model_file, ex_args=args) 
    parser.eval()
    cache = SynthCache.from_file(args.cache_file)

    synthesizer = EnumSynthesizer(args, parser, score_func='ubound')
    # synthesizer = NoPruneSynthesizer(args, parser, score_func='prob')
    # with torch.no_grad():
    synth_results = []
    budgets_used = []
    times_used = []
    init_backend()
    for ex in tqdm(test_set, desc='Synthesize', file=sys.stdout, total=len(test_set)):
        result, num_exec, time = synthesizer.solve(ex, cache=cache)
        synth_results.append(result)
        budgets_used.append(num_exec)
        times_used.append(time)
    exit_backend()
    act_tree_to_ast = lambda x: parser.transition_system.build_ast_from_actions(x)
    pred_codes = [[parser.transition_system.ast_to_surface_code(act_tree_to_ast(x.action_tree)) for x in preds] for preds in synth_results]
    top_codes = [x[0] if x else "" for x in pred_codes]
    match_results = [ " ".join(e.tgt_toks) == r for e, r in zip(test_set, top_codes)]
    match_acc = sum(match_results) * 1. / len(match_results)

    results = []
    acc = 0
    for pred_hyps, gt_exs in zip(pred_codes, test_set):
        # top_pred = pred_hyps[0]
        codes = [x.replace(" ", "") for x in pred_hyps]
        gt_code = " ".join(gt_exs.tgt_toks).replace(" ", "")

        match_result = batch_filtering_test(gt_code, codes, gt_exs.meta, flag_force=True)
        results.append(match_result)
        if match_result[0]:
            acc += 1
    cache.dump()
    print("Eval Acc", match_acc)
    print("Oracle Acc", acc * 1.0/len(test_set) )
    eval_results = [SynthResult(progs, budget, time, result) for (progs, budget, time, result) in zip(synth_results, budgets_used, times_used, results)]
    easy_pickle_dump(eval_results, args.eval_file)

    # if args.report_file:
    #     with open(args.report_file, "w") as f:s
    #         for i, res in enumerate(results):
    #             line_fields = [str(i), str(res[0]), str(res[1])]
    #             line_fields.extend(["{},{}".format(x[0], x[1])
    #                                 for x in enumerate(res[2])])
    #             f.write(" ".join(line_fields) + "\n")

    if args.decode_file:
        with open(args.decode_file, "w") as f:
            for preds in pred_codes:
                f.write(" ".join([c.replace(" ","") for c in preds])  + "\n")


def dump_graphvz(id, searched, prefix='gerror'):
    
    # node
    # wraped = ['node']
    filename = join(prefix, id + '.gv')
    make_dot_file(filename, searched)

    svg_file = join(prefix, id + '.svg')
    os.system('dot -Tsvg {} -o {}'.format(filename, svg_file))
    os.remove(filename)

def output_degug_info(args, debug_outputs, eval_results):
    # fname = 
    f = open('synth.debug.' + args.search_order + '.txt', 'w')
    for i, (verbose_info, eval_result) in enumerate(zip(debug_outputs, eval_results)):
        if verbose_info is None:
            continue

        result = verbose_info['result_node']
        explored = verbose_info['explored_nodes']

        # whether found
        if result is None:
            continue

        # whether prune actually happen
        num_prune_happened = sum([x.verify_result == False for x in explored])
        if not num_prune_happened:
            continue
        
        # let's print things
        print('---------------------------', file=f)
        print('{} {} {} {}, NumPrune {}'.format(str(i), str(eval_result[0]), eval_result[1], str(len(explored)), num_prune_happened), file=f)
        print('---------------------------', file=f)
        optimal_path = []
        cursor = result
        while cursor is not None:
            optimal_path.append(cursor.verify_time)
            cursor = cursor.parent

        for node in explored:
            # sibs = [x.verify_time for x in node.siblings]
            # sibs
            # sib_str = '[{}]'.format(' '.join(map(str,sibs)))
            print('T: {}, parent: {}, score: {:.3f}, OnPath: {}, Feasible: {}\n\t{}'.format(
                node.verify_time,
                node.create_time,
                node.hypothesis.score,
                node.verify_time in optimal_path,
                node.verify_result,
                node.partial_ast.debug_form() if node.partial_ast else 'None'
            ), file=f)

        print('\n', file=f)
        dump_graphvz('{}-{}'.format(args.search_order, i), explored)
    f.close()
        
def synth_debug(args):
    test_set = Dataset.from_bin_file(args.test_file)
    # poi = [36, 38, 40, 64, 69, 70, 82, 84, 89]
    # test_set.examples = [test_set.examples[i] for i in poi]
    test_set.examples = test_set.examples[89:90]
    parser = ASNParser.load(args.model_file, ex_args=args) 
    parser.eval()
    cache = SynthCache.from_file(args.cache_file)
    locator = easy_pickle_read(args.locator_file)

    synthesizer = LearnableSynthesizer(args, parser, locator, score_func='prob')
    # with torch.no_grad():
    synth_results = []
    budgets_used = []
    debug_outputs = []
    for ex in tqdm(test_set, desc='Synthesize', file=sys.stdout, total=len(test_set)):
        result, num_exec, verbose_info = synthesizer.debug(ex, cache=cache)
        synth_results.append(result)
        budgets_used.append(num_exec)

        # for memory's sake
        if verbose_info['result_node'] is None:
            debug_outputs.append(None)
        elif sum([x.verify_result == False for x in verbose_info['explored_nodes']]) == 0:
            debug_outputs.append(None)
        else:
            debug_outputs.append(verbose_info)
    act_tree_to_ast = lambda x: parser.transition_system.build_ast_from_actions(x)
    pred_codes = [[parser.transition_system.ast_to_surface_code(act_tree_to_ast(x.action_tree)) for x in preds] for preds in synth_results]
    top_codes = [x[0] if x else "" for x in pred_codes]
    match_results = [ " ".join(e.tgt_toks) == r for e, r in zip(test_set, top_codes)]
    match_acc = sum(match_results) * 1. / len(match_results)

    results = []
    acc = 0
    for pred_hyps, gt_exs in zip(pred_codes, test_set):
        # top_pred = pred_hyps[0]
        codes = [x.replace(" ", "") for x in pred_hyps]
        gt_code = " ".join(gt_exs.tgt_toks).replace(" ", "")

        match_result = batch_filtering_test(gt_code, codes, gt_exs.meta, flag_force=True)
        results.append(match_result)
        if match_result[0]:
            acc += 1
    # cache.dump()
    # print("Eval Acc", match_acc)
    # print("Oracle Acc", acc * 1.0/len(test_set) )

    # eval_results = [SynthResult(progs, budget, result) for (progs, budget, result) in zip(synth_results, budgets_used, results)]
    # easy_pickle_dump(eval_results, args.eval_file)

    output_degug_info(args, debug_outputs, results)
    # if args.report_file:
    #     with open(args.report_file, "w") as f:
    #         for i, res in enumerate(results):
    #             line_fields = [str(i), str(res[0]), str(res[1])]
    #             line_fields.extend(["{},{}".format(x[0], x[1])
    #                                 for x in enumerate(res[2])])
    #             f.write(" ".join(line_fields) + "\n")

    # if args.decode_file:
    #     with open(args.decode_file, "w") as f:
    #         for preds in pred_codes:
    #             f.write(" ".join([c.replace(" ","") for c in preds])  + "\n")

    

def synth_augment(args):
    test_set = Dataset.from_bin_file(args.test_file)
    parser = ASNParser.load(args.model_file, ex_args=args) 
    parser.eval()
    cache = SynthCache.from_file(args.cache_file)

    synthesizer = EnumSynthesizer(args, parser)


    synth_results = []
    budgets_used = []
    synth_traces = []
    for ex in tqdm(test_set, desc='Synthesize', file=sys.stdout, total=len(test_set)):
        result, num_exec, verbose_info = synthesizer.trace(ex, cache=cache)
        synth_results.append(result)
        budgets_used.append(num_exec)
        synth_traces.append(verbose_info)

    act_tree_to_ast = lambda x: parser.transition_system.build_ast_from_actions(x)
    pred_codes = [[parser.transition_system.ast_to_surface_code(act_tree_to_ast(x.action_tree)) for x in preds] for preds in synth_results]
    top_codes = [x[0] if x else "" for x in pred_codes]
    match_results = [ " ".join(e.tgt_toks) == r for e, r in zip(test_set, top_codes)]
    match_acc = sum(match_results) * 1. / len(match_results)

    results = []
    acc = 0
    for pred_hyps, gt_exs in zip(pred_codes, test_set):
        # top_pred = pred_hyps[0]
        codes = [x.replace(" ", "") for x in pred_hyps]
        gt_code = " ".join(gt_exs.tgt_toks).replace(" ", "")

        match_result = batch_filtering_test(gt_code, codes, gt_exs.meta, flag_force=True)
        results.append(match_result)
        if match_result[0]:
            acc += 1

    cache.dump()
    print("Eval Acc", match_acc)
    print("Oracle Acc", acc * 1.0/len(test_set) )

    easy_pickle_dump(synth_traces, args.eval_file.replace('eval', 'trace'))

def learnable_synthesize(args):
    test_set = Dataset.from_bin_file(args.test_file)
    # poi = [36, 38, 40, 64, 69, 70, 82, 84, 89]
    # test_set.examples = [test_set.examples[i] for i in poi]
    parser = ASNParser.load(args.model_file, ex_args=args) 
    parser.eval()
    cache = SynthCache.from_file(args.cache_file)
    locator = easy_pickle_read(args.locator_file)

    synthesizer = LearnableSynthesizer(args, parser, locator, score_func='prob')
    # with torch.no_grad():
    synth_results = []
    budgets_used = []
    for ex in tqdm(test_set, desc='Synthesize', file=sys.stdout, total=len(test_set)):
        result, num_exec = synthesizer.solve(ex, cache=cache)
        synth_results.append(result)
        budgets_used.append(num_exec)
    act_tree_to_ast = lambda x: parser.transition_system.build_ast_from_actions(x)
    pred_codes = [[parser.transition_system.ast_to_surface_code(act_tree_to_ast(x.action_tree)) for x in preds] for preds in synth_results]
    top_codes = [x[0] if x else "" for x in pred_codes]
    match_results = [ " ".join(e.tgt_toks) == r for e, r in zip(test_set, top_codes)]
    match_acc = sum(match_results) * 1. / len(match_results)

    results = []
    acc = 0
    for pred_hyps, gt_exs in zip(pred_codes, test_set):
        # top_pred = pred_hyps[0]
        codes = [x.replace(" ", "") for x in pred_hyps]
        gt_code = " ".join(gt_exs.tgt_toks).replace(" ", "")

        match_result = batch_filtering_test(gt_code, codes, gt_exs.meta, flag_force=True)
        results.append(match_result)
        if match_result[0]:
            acc += 1
    cache.dump()
    print("Eval Acc", match_acc)
    print("Oracle Acc", acc * 1.0/len(test_set) )

    eval_results = [SynthResult(progs, budget, result) for (progs, budget, result) in zip(synth_results, budgets_used, results)]
    easy_pickle_dump(eval_results, args.eval_file)

    if args.report_file:
        with open(args.report_file, "w") as f:
            for i, res in enumerate(results):
                line_fields = [str(i), str(res[0]), str(res[1])]
                line_fields.extend(["{},{}".format(x[0], x[1])
                                    for x in enumerate(res[2])])
                f.write(" ".join(line_fields) + "\n")

    if args.decode_file:
        with open(args.decode_file, "w") as f:
            for preds in pred_codes:
                f.write(" ".join([c.replace(" ","") for c in preds])  + "\n")


if __name__ == '__main__':
    args = parse_args('synthesize')
    synthesize(args)
    # learnable_synthesize(args)
    # synth_debug(args)
    # synth_augment(args)
