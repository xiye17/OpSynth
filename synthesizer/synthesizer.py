# from queue import PriorityQueue
from heapq import heappush, heappop, heapify
from grammar.transition_system import ApplyRuleAction, GenTokenAction, PlaceHolderAction, ActionTree
from grammar.hypothesis import Hypothesis
from .graph_viz import consolidate_tree
from grammar.streg.streg_transition_system import partial_asdl_ast_to_streg_ast, preverify_regex_with_exs, batch_preverify_regex_with_exs, asdl_ast_to_streg_ast, is_equal_ast, is_partial_ast
from .features import *

import time
import math
import numpy as np

class SearchTreeNode:
    # ontrack, offtrack'
    # t time out popup
    # def __init__(self, hypothesis, parent, children, create_time, access_time, siblings, idx_among_siblings):
    def __init__(self, hypothesis, create_time, parent):
        self.hypothesis = hypothesis
        self.create_time = create_time
        self.parent = parent
        self.children = []

        self.verify_time = -1
        self.verify_result = None
        self.partial_ast = None
        
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
        # self.siblings = []
        # self.idx_among_siblings = -1

    def __lt__(self, other):
        # if not isinstance(other, SearchTreeNode):
        #     return NotImplemented
        return self.hypothesis.score < other.hypothesis.score

    def record_verificatioin_result(self, verify_time, result, partial_ast):
        self.verify_time = verify_time
        self.verify_result = result
        self.partial_ast = partial_ast

    # def set_siblings_info(self, siblings, idx_among_siblings):
    #     self.siblings = siblings
    #     self.idx_among_siblings = idx_among_siblings

    def set_children(self, children):
        self.children = children

class Timer:
    def __init__(self):
        self.start = .0

    def reset(self):
        self.start = time.time()
    
    def time(self):
        return time.time() - self.start

class BasicPriorityQueue:
    def __init__(self, max_size=20000):
        self.heap = []
        self.max_size = max_size
        # self.size = size
    
    def push(self, item):
        heappush(self.heap, item)
        # consolidate
        if len(self.heap) >= 2 * self.max_size:
            # print('consolidate')
            # sanity check
            # for i, (s0, h0) in enumerate(self.heap):
            #     for s1, h1 in self.heap[i + 1:]:
            #         if s0 == s1:
            #             print(s0, s1)
            #             print(h0.action_tree)
            #             print(h1.action_tree)
            # print('resize')
            new_heap = []
            for _ in range(self.max_size):
                heappush(new_heap, self.pop())
            self.heap = new_heap

    def pop(self):
        return heappop(self.heap)
    
    def resort(self):
        heapify(self.heap)
    
    def __len__(self):
        return len(self.heap)


class EnumSynthesizer:
    def __init__(self, args, parser, score_func='prob'):
        # max_prog_size = 70
        # max_enum_steps = 0
        # max_enum_time = 0
        self.args = args
        self.parser = parser
        self.order = args.search_order
        self.max_running_step = 5000
        self.score_func = score_func

    def solve(self, example, cache=None):
        result, num_budget, _, time = self.trace(example, cache)
        return result, num_budget, time

    # def continuations_of_hyp(self, hyp, parser_state):
    #     progs = self.parser.continuations_of_hyp_for_synthesizer(hyp, parser_state)
    #     return [(-prog.score, prog) for prog in progs]

    def score_of_search_node(self, node):
        if self.score_func == 'prob':
            return -node.hypothesis.score
        elif self.score_func == 'astar':
            return -(node.hypothesis.score + 0.3 * node.depth)
        elif self.score_func == 'ubound':
            return -node.hypothesis.estimate_score_upperbound()

    def debug(self, example, cache=None):
        result, num_budget, explored_nodes = self.trace(example, cache)
        return result, num_budget, {'result_node': explored_nodes[-1], 'explored_nodes': explored_nodes}

    def trace(self, example, cache=None):
        worklist = BasicPriorityQueue()
        explored_nodes = []

        num_exec_steps = 0
        num_enumerated_progs = 0
        num_actual_check = 0


        init_hyp, parser_state = self.parser.initialize_hyp_for_synthesizing(example)
        init_node = SearchTreeNode(init_hyp, num_exec_steps, None)
        worklist.push((self.score_of_search_node(init_node), init_node))
        # num_enum_steps = 

        result = []

        timer = Timer()
        timer.reset()

        while num_exec_steps < self.max_running_step:
            # pick the top1 from worklist
            _, node = worklist.pop()

            explored_nodes.append(node)

            prog = node.hypothesis
            # prog is a hypothesis instance

            # check this partial prog
            # print(prog.action_tree)
            is_checkable, partial_ast = partial_asdl_ast_to_streg_ast(self.parser.transition_system.build_ast_from_actions(prog.action_tree))
            num_exec_steps += 1

            if is_checkable:
                num_actual_check += 1
                if prog.is_complete():
                    num_enumerated_progs += 1
                verify_result = preverify_regex_with_exs(partial_ast, example, cache=cache)
            else:
                verify_result = True
            node.record_verificatioin_result(num_exec_steps, verify_result, partial_ast)

            if not verify_result:
                continue
            # it will sure to pass the test

            if prog.is_complete():
                # test consistency
                result.append(prog)
                if len(result) == 1:
                    break
            else:
                # expand a prog
                # get continous of program, which is a set of new hypothesis
                # this should use a mixed strategy
                # key, continous
                continuations = self.continuations_of_node(node, parser_state, num_exec_steps)
                for c in continuations:
                    worklist.push(c)
        time_used = timer.time()
        print(len(result), 'budget used', num_exec_steps, 'actual check', num_actual_check, 'enum prog', num_enumerated_progs, 'time', time_used)
        # consolidate_tree(explored_nodes[0])
        return result, num_exec_steps, explored_nodes, time_used

    def continuations_of_node(self, node, parser_state, exec_time):
        hyp = node.hypothesis
        progs = self.parser.continuations_of_hyp_for_synthesizer(hyp, parser_state)
        # rank siblings in the descending order of scores
        # progs.sort(key=lambda x: -x.score)
        
        # for i, p in enumerate(progs):
        new_nodes = [SearchTreeNode(p, exec_time, node) for p in progs]
        # node.set_children(new_nodes)


        # for i, new_n in enumerate(new_nodes):
        #     new_n.set_siblings_info(new_nodes, i)

        return [(self.score_of_search_node(n), n) for n in new_nodes]

class NoPruneSynthesizer(EnumSynthesizer):
    def trace(self, example, cache=None):
        worklist = BasicPriorityQueue()
        explored_nodes = []

        num_exec_steps = 0
        num_enumerated_progs = 0
        num_actual_check = 0


        init_hyp, parser_state = self.parser.initialize_hyp_for_synthesizing(example)
        init_node = SearchTreeNode(init_hyp, num_exec_steps, None)
        worklist.push((self.score_of_search_node(init_node), init_node))
        # num_enum_steps = 

        result = []

        timer = Timer()
        timer.reset()

        while num_exec_steps < self.max_running_step:
            # pick the top1 from worklist
            if len(worklist) == 0:
                break
            _, node = worklist.pop()

            explored_nodes.append(node)

            prog = node.hypothesis
            # prog is a hypothesis instance
            num_exec_steps += 1
            
            if prog.is_complete():
                _, partial_ast = partial_asdl_ast_to_streg_ast(self.parser.transition_system.build_ast_from_actions(prog.action_tree))
                # print(num_exec_steps, partial_ast.debug_form())
                num_actual_check += 1
                num_enumerated_progs += 1
                verify_result = preverify_regex_with_exs(partial_ast, example, cache=cache)
            else:
                verify_result = True

            if not verify_result:
                continue
            # it will sure to pass the test

            if prog.is_complete():
                # test consistency
                result.append(prog)
                if len(result) == 1:
                    break
            else:
                # expand a prog
                # get continous of program, which is a set of new hypothesis
                # this should use a mixed strategy
                # key, continous
                continuations = self.continuations_of_node(node, parser_state, num_exec_steps)
                for c in continuations:
                    worklist.push(c)
        time_used = timer.time()
        print(len(result), 'budget used', num_exec_steps, 'actual check', num_actual_check, 'enum prog', num_enumerated_progs, 'time', time_used)
        consolidate_tree(explored_nodes[0])        
        return result, num_exec_steps, explored_nodes, time_used

class MutableTreeNode:
    # ontrack, offtrack'
    # t time out popup
    # def __init__(self, hypothesis, parent, children, create_time, access_time, siblings, idx_among_siblings):
    def __init__(self, hypothesis, create_time, parent):
        self.hypothesis = hypothesis
        self.create_time = create_time
        self.parent = parent
        self.children = []

        self.verify_time = -1
        self.verify_result = None
        self.partial_ast = None
        self.flaged = False
        self.increased = False
        self.decreased = False

        if parent is None:
            self.depth = 0
            self.action_score = self.hypothesis.score
        else:
            self.depth = parent.depth + 1
            self.action_score = self.hypothesis.score - parent.hypothesis.score
        self.siblings = []
        self.idx_among_siblings = -1
        self.sibling_probs = []
        self.search_score = .0

    def __lt__(self, other):
        if not isinstance(other, MutableTreeNode):
            return NotImplemented
        # larger the search score, more priviliage
        return (-self.search_score) < (-other.search_score)

    def record_verificatioin_result(self, verify_time, result, partial_ast):
        self.verify_time = verify_time
        self.verify_result = result
        self.partial_ast = partial_ast

    def set_siblings_info(self, siblings, idx_among_siblings, sibling_probs):
        self.siblings = siblings
        self.idx_among_siblings = idx_among_siblings
        self.sibling_probs = sibling_probs

    def set_children(self, children):
        self.children = children


class FeedbackSynthesizer:
    def __init__(self, args, parser, score_func='prob'):
        # max_prog_size = 70
        # max_enum_steps = 0
        # max_enum_time = 0
        self.args = args
        self.parser = parser
        self.order = args.search_order
        self.max_running_step = 1500
        self.score_func = score_func

    def solve(self, example, cache=None):
        result, num_budget, _ = self.trace(example, cache)
        return result, num_budget

    def score_of_search_node(self, node):
        if node.parent is None:
            node.search_score = node.hypothesis.score 
        else:
            node.search_score = node.parent.search_score + node.action_score
            # print(node.search_score)

    def debug(self, example, cache=None):
        result, num_budget, explored_nodes = self.trace(example, cache)
        return result, num_budget, {'result_node': explored_nodes[-1], 'explored_nodes': explored_nodes}

    def trace(self, example, cache=None):
        worklist = BasicPriorityQueue()
        explored_nodes = []

        num_exec_steps = 0
        num_enumerated_progs = 0
        num_actual_check = 0


        init_hyp, parser_state = self.parser.initialize_hyp_for_synthesizing(example)
        init_node = MutableTreeNode(init_hyp, num_exec_steps, None)
        self.score_of_search_node(init_node)
        worklist.push(init_node)
        # num_enum_steps = 

        result = []

        timer = Timer()
        timer.reset()

        while num_exec_steps < self.max_running_step:
            # pick the top1 from worklist
            node = worklist.pop()

            explored_nodes.append(node)

            prog = node.hypothesis
            # prog is a hypothesis instance

            # check this partial prog
            # print(prog.action_tree)
            is_checkable, partial_ast = partial_asdl_ast_to_streg_ast(self.parser.transition_system.build_ast_from_actions(prog.action_tree))
            num_exec_steps += 1

            if is_checkable:
                num_actual_check += 1
                if prog.is_complete():
                    num_enumerated_progs += 1
                verify_result = preverify_regex_with_exs(partial_ast, example, cache=cache)
            else:
                verify_result = True
            node.record_verificatioin_result(num_exec_steps, verify_result, partial_ast)

            if not verify_result:
                self.reweight_by_error(node)
                worklist.resort()
                continue

            # it will sure to pass the test

            if prog.is_complete():
                # test consistency
                result.append(prog)
                if len(result) == 1:
                    break
            else:
                # expand a prog
                # get continous of program, which is a set of new hypothesis
                # this should use a mixed strategy
                # key, continous
                continuations = self.continuations_of_node(node, parser_state, num_exec_steps)
                for c in continuations:
                    worklist.push(c)

        print(len(result), 'budget used', num_exec_steps, 'actual check', num_actual_check, 'enum prog', num_enumerated_progs, 'time', timer.time())
        consolidate_tree(explored_nodes[0])
        return result, num_exec_steps, explored_nodes

    def continuations_of_node(self, node, parser_state, exec_time):
        hyp = node.hypothesis
        progs = self.parser.continuations_of_hyp_for_synthesizer(hyp, parser_state)
        # rank siblings in the descending order of scores
        progs.sort(key=lambda x: -x.score)
        
        # for i, p in enumerate(progs):
        new_nodes = [MutableTreeNode(p, exec_time, node) for p in progs]
        node.set_children(new_nodes)
        sibling_probs = np.array([ n.action_score for n in new_nodes])
        sibling_probs = np.exp(sibling_probs)
        for i, new_n in enumerate(new_nodes):
            new_n.set_siblings_info(new_nodes, i, sibling_probs)
        [self.score_of_search_node(n) for n in new_nodes]
        return new_nodes
    
    def trace_back_to_root(self, node):
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

    def error_indicator(self, node):
        ind_type = 'init'
        if ind_type == 'init':
            return node.hypothesis.score - node.parent.hypothesis.score
        if ind_type == 'naive':
            return node.action_score
        elif ind_type == 'scale':
            sib_probs = node.sibling_probs 
            still_mask = np.asarray([x.flaged for x in node.siblings])
            cur_idx = node.idx_among_siblings
            cur_prob = sib_probs[cur_idx]
            return cur_prob / (1 - 0.5 * np.sum(sib_probs[still_mask]))

    def reweight_by_error(self, node):
        # path
        path = self.trace_back_to_root(node)
        path.sort(key=lambda x: self.error_indicator(x))

        # bug
        bug_node = path[0]
        self.upweight_siblings_of_bug(bug_node)
        

    def _upweight_siblings_of_bug(self, bug_node):
        # multiple strategies
        # strictly probalistic

        # only perturb search scores
        # if bug_node.flaged:
        #     return
        
        bug_node.flaged = True
        
        for i, s in enumerate(bug_node.siblings):
            # if s.flaged:
            #     continue
            if i != bug_node.idx_among_siblings:
                s.search_score += 0.25
            self.notify_score_change(s) 

    def upweight_siblings_of_bug(self, bug_node):
        strategy = 'B'
        # strategy A continue on decreasing
        if strategy == 'A':
            pass

        # strategy B decrease once
        elif strategy == 'B':
            if bug_node.flaged:
                return
            bug_node.flaged = True
            # decrease probs of bug_node
            cur_idx = bug_node.idx_among_siblings
            dec_factor = 0.75
            sibling_probs = bug_node.sibling_probs
            sibling_probs[cur_idx] *= dec_factor
            # stil_mask
            # still_mask = np.asarray([x.flaged for x in bug_node.siblings])
            still_mask = np.asarray([cur_idx == i for i in range(len(bug_node.siblings))])
            sum_probs_still = np.sum(sibling_probs[still_mask])
            sibling_probs[~still_mask] =  sibling_probs[~still_mask] * (1 - sum_probs_still) / np.sum(sibling_probs[~still_mask])
            new_ac_scores = np.log(sibling_probs)
            # inplace modification, keep everything on the same page
            for i, s in enumerate(bug_node.siblings):
                if still_mask[i] and (i != cur_idx):
                    continue
                s.action_score = new_ac_scores[i]
                s.search_score = s.parent.search_score + s.action_score
                self.notify_score_change(s)
        else:
            pass

    def notify_score_change(self, node):
        for c in node.children:
            c.search_score = node.search_score + c.action_score
            self.notify_score_change(c)



class LearnableSynthesizer:
    def __init__(self, args, parser, locator, score_func='prob'):
        # max_prog_size = 70
        # max_enum_steps = 0
        # max_enum_time = 0
        self.args = args
        self.parser = parser
        self.order = args.search_order
        self.max_running_step = 1500
        self.locator = locator
        self.score_func = score_func

    def solve(self, example, cache=None):
        result, num_budget, _ = self.trace(example, cache)
        return result, num_budget

    def score_of_search_node(self, node):
        if node.parent is None:
            node.search_score = node.hypothesis.score 
        else:
            node.search_score = node.parent.search_score + node.action_score
            # print(node.search_score)

    def debug(self, example, cache=None):
        result, num_budget, explored_nodes = self.trace(example, cache)
        return result, num_budget, {'result_node': explored_nodes[-1], 'explored_nodes': explored_nodes}

    def trace(self, example, cache=None):
        worklist = BasicPriorityQueue()
        explored_nodes = []

        num_exec_steps = 0
        num_enumerated_progs = 0
        num_actual_check = 0
        num_fails = 0

        init_hyp, parser_state = self.parser.initialize_hyp_for_synthesizing(example)
        init_node = MutableTreeNode(init_hyp, num_exec_steps, None)
        self.score_of_search_node(init_node)
        worklist.push(init_node)
        # num_enum_steps = 

        result = []

        timer = Timer()
        timer.reset()

        while num_exec_steps < self.max_running_step:
            # pick the top1 from worklist
            node = worklist.pop()

            explored_nodes.append(node)

            prog = node.hypothesis
            # prog is a hypothesis instance

            # check this partial prog
            # print(prog.action_tree)
            is_checkable, partial_ast = partial_asdl_ast_to_streg_ast(self.parser.transition_system.build_ast_from_actions(prog.action_tree))
            num_exec_steps += 1

            if is_checkable:
                num_actual_check += 1
                if prog.is_complete():
                    num_enumerated_progs += 1
                verify_result = preverify_regex_with_exs(partial_ast, example, cache=cache)
            else:
                verify_result = True
            node.record_verificatioin_result(num_exec_steps, verify_result, partial_ast)

            if not verify_result:
                num_fails += 1
                self.reweight_by_error(node, num_fails)
                worklist.resort()
                continue

            # it will sure to pass the test

            if prog.is_complete():
                # test consistency
                result.append(prog)
                if len(result) == 1:
                    break
            else:
                # expand a prog
                # get continous of program, which is a set of new hypothesis
                # this should use a mixed strategy
                # key, continous
                continuations = self.continuations_of_node(node, parser_state, num_exec_steps)
                for c in continuations:
                    worklist.push(c)

        print(len(result), 'budget used', num_exec_steps, 'actual check', num_actual_check, 'enum prog', num_enumerated_progs, 'time', timer.time())
        consolidate_tree(explored_nodes[0])
        return result, num_exec_steps, explored_nodes

    def continuations_of_node(self, node, parser_state, exec_time):
        hyp = node.hypothesis
        progs = self.parser.continuations_of_hyp_for_synthesizer(hyp, parser_state)
        # rank siblings in the descending order of scores
        progs.sort(key=lambda x: -x.score)
        
        # for i, p in enumerate(progs):
        new_nodes = [MutableTreeNode(p, exec_time, node) for p in progs]
        node.set_children(new_nodes)
        sibling_probs = np.array([ n.action_score for n in new_nodes])
        sibling_probs = np.exp(sibling_probs)
        for i, new_n in enumerate(new_nodes):
            new_n.set_siblings_info(new_nodes, i, sibling_probs)
        [self.score_of_search_node(n) for n in new_nodes]
        return new_nodes
    
    def trace_back_to_root(self, node):
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

    def locate_error(self, path):
        branch = make_branch_info(path)
        features = extract_features_for_branch(branch)
        predictions = predict_error(self.locator, branch, features)
        max_idx = np.argmax(predictions)
        
        return max_idx, predictions

    def reweight_by_error(self, node, num_fails):
        # path
        # moderate modification
        if num_fails > 1:
            return
        path = self.trace_back_to_root(node)
        # bug

        usage = 'downweight'
        if usage == 'downweight':
            max_idx, predictions = self.locate_error(path)
            if predictions[max_idx] < 0.6:
                return
            bug_node = path[max_idx]

            # print('preds:', predictions)
            # for node in path:
            #     print(node.hypothesis.last_filled_node.action, end=' | ')        
            # print('\n', max_idx, path[max_idx].hypothesis.last_filled_node.action)

            self.upweight_siblings_of_bug(bug_node, predictions[max_idx])
        elif usage == 'comprehensive':
            max_idx, predictions = self.locate_error(path)
            max_pred_val = predictions[max_idx]
            # if predictions[max_idx] > 0.5:
            #     # print('do_downweight')
            #     self.downweight_node_choice(path[max_idx])
            
            sure_threshold = 0.05
            gap_threshold = 0.15
            gap_with_max = max_pred_val - predictions
            sured_entries = np.logical_and(predictions < sure_threshold, gap_with_max > gap_threshold)
            for e, node in zip(sured_entries, path):
                if e:
                    self.intensify_node_choice(node)
        elif usage == 'heuristic':
            overall_l = len(path)
            # scores = [n.action_score for n in path]
            max_dec = int(overall_l * 0.6)
            for i, node in enumerate(path[:max_dec]):
                if node.idx_among_siblings == 0:
                    factor = 4.5 - 0.5 * i
                    if factor < 1:
                        continue
                    self.peakify_node_choice(node, factor)

    def peakify_node_choice(self, bug_node, inc_factor=3.5):
        if bug_node.increased:
            return
        bug_node.increased = True
        # decrease probs of bug_node
        cur_idx = bug_node.idx_among_siblings
        sibling_probs = bug_node.sibling_probs
        sibling_probs[cur_idx] *= inc_factor
        sibling_probs *= (1 / np.sum(sibling_probs))
        new_ac_scores = np.log(sibling_probs)
        # inplace modification, keep everything on the same page
        for i, s in enumerate(bug_node.siblings):
            s.action_score = new_ac_scores[i]
            s.search_score = s.parent.search_score + s.action_score
            self.notify_score_change(s)
    
    def intensify_node_choice(self, bug_node):
        # if bug_node.increased:
        #     return
        if bug_node.action_score > -0.001:
            return
        print('Intensify\t', end='')
        bug_node.increased = True
        # decrease probs of bug_node
        cur_idx = bug_node.idx_among_siblings
        inc_factor = 3.5
        sibling_probs = bug_node.sibling_probs
        # print(bug_node.verify_time, sibling_probs, cur_idx)
        sibling_probs[cur_idx] *= inc_factor
        sibling_probs *= (1 / np.sum(sibling_probs))
        # print(bug_node.verify_time, sibling_probs)
        new_ac_scores = np.log(sibling_probs)
        # inplace modification, keep everything on the same page
        for i, s in enumerate(bug_node.siblings):
            s.action_score = new_ac_scores[i]
            s.search_score = s.parent.search_score + s.action_score
            self.notify_score_change(s)

    def downweight_node_choice(self, bug_node):
        if bug_node.decreased:
            return
        bug_node.decreased = True
        cur_idx = bug_node.idx_among_siblings
        dec_factor = 0.5
        sibling_probs = bug_node.sibling_probs
        sibling_probs[cur_idx] *= dec_factor
        # stil_mask
        still_mask = np.asarray([x.decreased for x in bug_node.siblings])
        # still_mask = np.asarray([cur_idx == i for i in range(len(bug_node.siblings))])

        sum_probs_still = np.sum(sibling_probs[still_mask])
        sibling_probs[~still_mask] =  sibling_probs[~still_mask] * (1 - sum_probs_still) / np.sum(sibling_probs[~still_mask])
        new_ac_scores = np.log(sibling_probs)

        # inplace modification, keep everything on the same page
        for i, s in enumerate(bug_node.siblings):
            if still_mask[i] and (i != cur_idx):
                continue
            s.action_score = new_ac_scores[i]
            s.search_score = s.parent.search_score + s.action_score
            self.notify_score_change(s)

    def upweight_siblings_of_bug(self, bug_node, bug_prob):
        # multiple strategies
        # strictly probalistic

        # only perturb search scores
        # if bug_node.flaged:
        #     return
        
        bug_node.flaged = True
        for i, s in enumerate(bug_node.siblings):
            # if s.flaged:
            #     continue
            if i != bug_node.idx_among_siblings:
                s.search_score += 1.5
            self.notify_score_change(s)

    def _upweight_siblings_of_bug(self, bug_node, bug_prob):
        strategy = 'B'
        # strategy A continue on decreasing
        # strategy = 'C'
        if strategy == 'A':
            if bug_node.flaged:
                return
            bug_node.flaged = True
            # decrease probs of bug_node
            cur_idx = bug_node.idx_among_siblings
            dec_factor = 0.5
            sibling_probs = bug_node.sibling_probs
            sibling_probs[cur_idx] *= dec_factor

            bug_node.action_score = bug_node.action_score + math.log(dec_factor)
            bug_node.search_score = bug_node.parent.search_score + bug_node.action_score
            # stil_mask
            self.notify_score_change(bug_node)

        # strategy B decrease once
        elif strategy == 'B':
            if bug_node.flaged:
                return
            bug_node.flaged = True
            # decrease probs of bug_node
            cur_idx = bug_node.idx_among_siblings
            dec_factor = 1 - bug_prob
            sibling_probs = bug_node.sibling_probs
            sibling_probs[cur_idx] *= dec_factor
            # stil_mask
            still_mask = np.asarray([x.flaged for x in bug_node.siblings])
            # still_mask = np.asarray([cur_idx == i for i in range(len(bug_node.siblings))])
            sum_probs_still = np.sum(sibling_probs[still_mask])
            sibling_probs[~still_mask] =  sibling_probs[~still_mask] * (1 - sum_probs_still) / np.sum(sibling_probs[~still_mask])
            new_ac_scores = np.log(sibling_probs)
            # inplace modification, keep everything on the same page
            for i, s in enumerate(bug_node.siblings):
                if still_mask[i] and (i != cur_idx):
                    continue
                s.action_score = new_ac_scores[i]
                s.search_score = s.parent.search_score + s.action_score
                self.notify_score_change(s)

        elif strategy == 'C':
            # if bug_node.flaged:
            #     return
            bug_node.flaged = True
            # decrease probs of bug_node
            cur_idx = bug_node.idx_among_siblings
            dec_factor = 1 - bug_prob
            sibling_probs = bug_node.sibling_probs
            sibling_probs[cur_idx] *= dec_factor
            # stil_mask
            # still_mask = np.asarray([x.flaged or i < cur_idx for i, x in enumerate(bug_node.siblings)])
            still_mask = np.asarray([x.flaged for i, x in enumerate(bug_node.siblings)])
            # still_mask = np.asarray([cur_idx == i for i in range(len(bug_node.siblings))])
            sum_probs_still = np.sum(sibling_probs[still_mask])
            sibling_probs[~still_mask] =  sibling_probs[~still_mask] * (1 - sum_probs_still) / np.sum(sibling_probs[~still_mask])
            # print(sibling_probs)
            new_ac_scores = np.log(sibling_probs)
            # inplace modification, keep everything on the same page
            for i, s in enumerate(bug_node.siblings):
                if still_mask[i] and (i != cur_idx):
                    continue
                s.action_score = new_ac_scores[i]
                s.search_score = s.parent.search_score + s.action_score
                self.notify_score_change(s)

    def notify_score_change(self, node):
        for c in node.children:
            c.search_score = node.search_score + c.action_score
            self.notify_score_change(c)
