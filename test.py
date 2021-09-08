from common.config import *
from components.dataset import *
from components.cache import *
from components.result import DecodeResult
from grammar.grammar import Grammar
from common.utils import *

# from grammar.streg.streg_transition_system import 
from models.ASN import ASNParser
from models import nn_utils

from torch import optim
import os
from os.path import join
import random

import subprocess
from tqdm import tqdm
import sys

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

def test(args):
    test_set = Dataset.from_bin_file(args.test_file)
    parser = ASNParser.load(args.model_file, ex_args=args) 
    parser.eval()

    if (not args.do_naive) and (not args.do_filter):
        cache = SynthCache.from_file(args.cache_file)
    with torch.no_grad():
        parse_results = []
        for ex in tqdm(test_set, desc='Decoding', file=sys.stdout, total=len(test_set)):
            if args.do_naive:
                parse_results.append(parser.naive_parse(ex))
            elif args.do_filter:
                parse_results.append(parser.parse(ex))
            else:
                parse_results.append(parser.ex_guided_parse(ex, cache=cache))

    budgets = []
    if (not args.do_naive) and (not args.do_filter):
        budgets = [x[1] for x in parse_results]
        parse_results = [x[0] for x in parse_results]
    act_tree_to_ast = lambda x: parser.transition_system.build_ast_from_actions(x)

    if args.do_naive:
        match_results = [parser.transition_system.compare_ast(e.tgt_ast, r) for e, r in zip(test_set, parse_results)]
        match_acc = sum(match_results) * 1. / len(match_results)
        print("Eval Acc", match_acc)
        return

    pred_codes = [[parser.transition_system.ast_to_surface_code(act_tree_to_ast(x.action_tree)) for x in preds] for preds in parse_results]
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
    if (not args.do_naive) and (not args.do_filter):
        cache.dump()
    print("Eval Acc", match_acc)
    print("Oracle Acc", acc * 1.0/len(test_set) )

    if args.report_file:
        with open(args.report_file, "w") as f:
            for i, res in enumerate(results):
                line_fields = [str(i), str(res[0]), str(res[1])]
                line_fields.extend(["{},{}".format(x[0], x[1])
                                    for x in enumerate(res[2])])
                f.write(" ".join(line_fields) + "\n")

    # if args.decode_file:
    #     with open(args.decode_file, "w") as f:
    #         for preds in pred_codes:
    #             f.write(" ".join([c.replace(" ","") for c in preds])  + "\n")
    if args.decode_file:
        if not budgets:
            decodes_result = [DecodeResult(a, b) for a, b in zip(parse_results, results)]
        else:
            decodes_result = [DecodeResult(a, b, c) for a, b, c in zip(parse_results, results, budgets)]
        easy_pickle_dump(decodes_result, args.decode_file)

if __name__ == '__main__':
    args = parse_args('test')
    test(args)
    # best_first_test('test')
