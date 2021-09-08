import pickle
from os.path import join

from synthesizer.synthesizer import SearchTreeNode
from components.dataset import *
from grammar.transition_system import PlaceHolderAction

import math

def stats_of_trace(split):
    synth_trace_file = join('outputs', 'synth.trace.model.streg.enc.src100.field100.drop0.3.max_ep100.batch32.lr0.003.clip_grad5.0.bin.%s.dfs.pkl' % split)    
    with open(synth_trace_file, 'rb') as f:
        synth_traces = pickle.load(f)
    
    # some key stats
    
    num_instance = len(synth_traces)
    
    # successed
    synth_traces = [trace for trace in synth_traces if trace[-1].verify_result and trace[-1].hypothesis.is_complete()]
    print('Num Succeed', len(synth_traces), len(synth_traces)/ num_instance)

    # num of pruned
    num_prune_times = [sum([x.verify_result == False for x in trace]) for trace in synth_traces]
    
    # no prune happen
    num_in_oneshot = sum([x == 0 for x in num_prune_times])
    print('Num In Oneshot', num_in_oneshot, num_in_oneshot / num_instance)

    num_pruned_once = sum([x == 1 for x in num_prune_times])
    print('Num Pruned Once', num_pruned_once, num_pruned_once / num_instance)
    
    num_multi_prune_times = [x for x in num_prune_times if x > 1]
    print('Num Pruned Multiple Times', len(num_multi_prune_times), len(num_multi_prune_times) / num_instance)
    # for u in unique_times:
    for lb, ub in [(2,5), (5,10), (10, 100), (100, 10000)]:
        num_pruned_u = sum([x >= lb and x < ub for x in num_prune_times])
        print('Pruned {} - {} Times, {}, {}'.format(lb, ub, num_pruned_u, num_pruned_u / num_instance))


def trace_back_to_root(node, trace):
    # cursor = node
    cursor = node
    # if it is none, it should be the root
    node_seq = []
    while cursor.hypothesis.last_filled_node is not None:
        node_seq.append(cursor)
        cursor = cursor.parent

    node_seq.reverse()
    # print( [(x.verify_time, x.create_time) for x in node_seq])
    return node_seq

def _linearize_action_tree(node, l, d, parent_ptrs, parent_id):
    if isinstance(node.action, PlaceHolderAction):
        return
    else:
        node.depth = d
        l.append(node)
        parent_ptrs.append(parent_id)
        p_idx = len(parent_ptrs) - 1
        for f in node.fields:
            _linearize_action_tree(f, l, d+1, parent_ptrs, p_idx)
def linearize_action_tree(hyp):
    root = hyp.action_tree
    nodes = []
    parent_ptrs = []
    _linearize_action_tree(root, nodes, 0, parent_ptrs, -1)
    return nodes, parent_ptrs

def make_branch_info(fail_seq, optimal_seq):

    final_hyp = fail_seq[-1].hypothesis
    node_seq, parents_ptr = linearize_action_tree(final_hyp)
    print(parents_ptr)
    # exit()
    assert len(node_seq) == len(fail_seq)
    action_seq = []
    i = 0
    while True:
        fn = fail_seq[i]
        gn = optimal_seq[i]
        action = fn.hypothesis.last_filled_node.action
        v_state = fn.parent.hypothesis.get_pending_node().action.v_state
        hole = fn.parent.hypothesis.get_pending_node().action
        score = fn.hypothesis.score - fn.parent.hypothesis.score
        if fn.verify_time != gn.verify_time:
            action_seq.append(SynthAction(v_state, action, 'error', score, hole, node_seq[i]))
            break
        else:
            action_seq.append(SynthAction(v_state, action, 'correct', score, hole, node_seq[i]))
        i += 1

    for j in range(i + 1, len(fail_seq)):
        fn = fail_seq[j]
        action = fn.hypothesis.last_filled_node.action
        v_state = fn.parent.hypothesis.get_pending_node().action.v_state
        score = fn.hypothesis.score - fn.parent.hypothesis.score
        hole = fn.parent.hypothesis.get_pending_node().action
        action_seq.append(SynthAction(v_state, action, 'irrelevent', score, hole, node_seq[i]))
    return SynthBranch(action_seq, final_hyp, parents_ptr)

def make_synth_dataset(split):
    synth_trace_file = join('outputs', 'synth.trace.model.streg.enc.src100.field100.drop0.3.max_ep100.batch32.lr0.003.clip_grad5.0.bin.%s.dfs.pkl' % split)    
    with open(synth_trace_file, 'rb') as f:
        synth_traces = pickle.load(f)

    paths = []
    for trace in synth_traces:
        end = trace[-1]
        
        if not (end.verify_result and end.hypothesis.is_complete()):
            paths.append(SynthPath([]))
            continue
        # terminated cursors
        optimal_seq = trace_back_to_root(end, trace)
        # terminated_points = []
        terminated_nodes = [x for x in trace if x.verify_result == False]
        branches = []
        for t_node in terminated_nodes:
            t_seq = trace_back_to_root(t_node, trace)
            branches.append(make_branch_info(t_seq, optimal_seq))
        
        paths.append(SynthPath(branches))
    
    pickle.dump(paths, open('data/streg/%s_pathaux.bin' % split, 'wb'))

def _recover_depth_info(node, depth, l):
    if isinstance(node.action, PlaceHolderAction):
        return
    else:
        l.append(depth)
        for f in node.fields:
            _recover_depth_info(f, depth + 1, l)

def recover_depth_info(branch):
    hypothesis = branch.hypothesis
    root = hypothesis.action_tree
    depth_info = []
    _recover_depth_info(root, 0, depth_info)
    # print(depth_info)
    # print([x.action for x in branch.actions])
    assert(len(depth_info) == len(branch.actions))
    return depth_info



# def _recover_depth_info(node, depth, l):
#     if isinstance(node.action, PlaceHolderAction):
#         return
#     else:
#         if node.parent is None:
#             l.append('none')
#         else:
#             l.append(node.parent.action.production.constructor.name)
#         for f in node.fields:
#             _recover_depth_info(f, depth + 1, l)

# def recover_depth_info(branch):
#     hypothesis = branch.hypothesis
#     root = hypothesis.action_tree
#     depth_info = []
#     _recover_depth_info(root, 0, depth_info)
#     # print(depth_info)
#     # print([x.action for x in branch.actions])
#     assert(len(depth_info) == len(branch.actions))
#     return depth_info


def ECE(decision_info):
    num_bin = 10
    margin = 1. / num_bin
    acc = 0
    # print(decision_info)
    # print([x[1] for x in decision_info])
    for b in range(num_bin):
        b_info = [x for x in decision_info if (b * margin) <= x[1] < ((b + 1) * margin)]
        if not b_info:
            continue
        acc +=  (sum([x[2] for x in b_info]) - sum([x[1] for x in b_info]))
    acc = acc / len(decision_info)
    # print(sum([x[1] for x in decision_info])/len(decision_info))
    # print(sum([x[1] for x in decision_info])/len(decision_info))
    return acc

def check_synth_dataset(split):
    paths = pickle.load(open('data/streg/%s_path.bin' % split, 'rb'))
    # for p in paths:
    print(len([x for x in paths if x.branches]))
    # calibration stats
    # collect first branch
    first_branches = []
    for path in paths:
        if not path.branches:
            continue
        first_branch = path.branches[0]
        first_branches.append(first_branch)
    print(len(first_branches))
    # collected wanted information
    decision_info = []
    for branch in first_branches:
        depth_info  = recover_depth_info(branch)
        for d, action in zip(depth_info, branch.actions):
            decision_info.append((d, math.exp(action.score), action.label != 'error'))
    
    all_depth = [x[0] for x in decision_info]
    all_depth = sorted(list(set(all_depth)))
    for d in all_depth:
        d_decision_info = [x for x in decision_info if x[0] == d]
        error = ECE(d_decision_info)
        print(d, len(d_decision_info), error)

def main():
    # stats_of_trace('train')
    # stats_of_trace('teste')
    # make_synth_dataset('train')
    # check_synth_dataset('train')

    # make_synth_dataset('teste')
    make_synth_dataset('testi')
    # check_synth_dataset('teste')

if __name__ == '__main__':
    main()