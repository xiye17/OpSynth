from .grammar import *
from .dsl_ast import *
from .transition_system import *
import torch

class Hypothesis:
    def __init__(self, order):
        # unclosed nodes should be tuple of (node, input state)
        # self.unclosed_nodes = []
        # linearized action tree
        self.action_tree = None
        # self.ast = None
        self.score = .0
        self.order = order
        self.t = 0
        self.pending_nodes = []

        # self.priority_scores = []
        self.unprocessed_nodes = []
        
        self.last_filled_node = None

    def __lt__(self, other):
        if not isinstance(other, Hypothesis):
            return NotImplemented
        return self.score < other.score

    def get_unprocessed_nodes(self):
        return self.unprocessed_nodes

    def estimate_score_upperbound(self):
        return self.score + sum([x.action.max_candidate_score for x in self.pending_nodes])

    def update_priority_scores(self):
        for node in self.unprocessed_nodes:
            node.action.flag_cached = True
            if self.order == 'entropy':
                scores = node.action.candidate_scores
                # minus entropy the higher the better
                priority = torch.sum(scores * torch.exp(scores)).item()
                node.action.priority = priority
            elif self.order == 'uncertain':
                scores = node.action.candidate_scores
                # minus entropy the higher the better
                priority = - torch.sum(scores * torch.exp(scores)).item()
                node.action.priority = priority
            if self.order == 'rule':
                if node.action.type.is_composite_type():
                    choice_idx = torch.argmax(node.action.candidate_scores).item()
                    choice = node.action.candidates[choice_idx].constructor.name
                    if choice in ['Optional', 'NotCC']:
                        priority = 0.0
                        # print('changed', choice)
                    else:
                        priority = 1.0
                else:
                    priority = 1.0
                node.action.priority = priority
        self.unprocessed_nodes = []

        # minus entropy from high to low
        if self.order == 'entropy':
            self.pending_nodes.sort(key=lambda x: x.action.priority, reverse=True)
        elif self.order == 'uncertain':
            self.pending_nodes.sort(key=lambda x: x.action.priority, reverse=True)
        elif self.order == 'rule':
            self.pending_nodes.sort(key=lambda x: x.action.priority, reverse=True)

    def update_pending_node(self):
        # r = self.unclosed_nodes[0]
        # self.unclosed_nodes
        # return r
        if self.order == "dfs":
            self._pending_node_by_dfs(self.action_tree)
        elif self.order == "bfs":    
            self._pending_node_by_bfs(self.action_tree)
        elif self.order == "entropy":
            self._pending_node_by_entropy(self.action_tree)
        elif self.order == "uncertain":
            self._pending_node_by_uncertain(self.action_tree)
        elif self.order == "rule":
            self._pending_node_by_rule(self.action_tree)
        else:
            raise ValueError('Invalid order of tranversing')

    def get_pending_node(self):
        node = self.pending_nodes[0]
        return self.pending_nodes[0]

    def _pending_node_by_dfs(self, node):
        if isinstance(node.action, PlaceHolderAction):
            self.pending_nodes.append(node)
        for x in node.fields:
            self._pending_node_by_dfs(x)

    def _pending_node_by_bfs(self, node):
        search_queue = [node]
        while search_queue:
            cur = search_queue[0]
            if isinstance(cur.action, PlaceHolderAction):
                self.pending_nodes.append(cur)

            search_queue = search_queue[1:]
            search_queue.extend(cur.fields)

    def _pending_node_by_entropy(self, node):
        self._pending_node_by_dfs(node)
        self.pending_nodes.sort(key=lambda x: x.action.priority, reverse=True)
    
    def _pending_node_by_uncertain(self, node):
        self._pending_node_by_dfs(node)
        self.pending_nodes.sort(key=lambda x: x.action.priority, reverse=True)
    

    # must keep things in the same order, after pending, because we'll pop node then
    def _pending_node_by_rule(self, node):
        self._pending_node_by_dfs(node)
        self.pending_nodes.sort(key=lambda x: x.action.priority, reverse=True)

    def copy_and_apply_action(self, action, score, updated_states=None):
        new_hyp = self.clone()
        new_hyp.t += 1
        new_hyp.apply_action(action, score, updated_states)
        return new_hyp

    def apply_action(self, action, score, updated_states=None):
        node = self.get_pending_node()
        node.action = action
        self.last_filled_node = node
        self.score = self.score + score
        self.pending_nodes = self.pending_nodes[1:]
        if isinstance(action, GenTokenAction):
            return 

        elif isinstance(action, ApplyRuleAction):
            assert updated_states is not None
            assert len(action.production.fields) == len(updated_states)
            constructor = action.production.constructor
            new_pending_nodes = [ActionTree(PlaceHolderAction(f.type, v_state), parent=node) for f, v_state in zip(constructor.fields, updated_states)]
            node.fields = new_pending_nodes
            
            self.unprocessed_nodes = new_pending_nodes
            if self.order == "dfs":
                self.pending_nodes = new_pending_nodes + self.pending_nodes
            elif self.order == "bfs":
                self.pending_nodes = self.pending_nodes + new_pending_nodes
            elif self.order == "entropy": 
                self.pending_nodes = self.pending_nodes + new_pending_nodes
            elif self.order == "uncertain": 
                self.pending_nodes = new_pending_nodes + self.pending_nodes
            elif self.order == "rule":
                self.pending_nodes = new_pending_nodes + self.pending_nodes
            else:
                raise ValueError('Invalid order of tranversing')
        else:
            raise ValueError("Invalid acction type")

    def clone(self):
        hyp = Hypothesis(self.order)
        hyp.action_tree = self.action_tree.copy()
        # hyp.ast = 
        hyp.score = self.score
        hyp.order = self.order
        hyp.t = self.t
        hyp.update_pending_node()
        return hyp

    @classmethod
    def init_hypothesis(cls, root_type, init_state, order):
        # node = ActionTree()
        node = ActionTree(PlaceHolderAction(root_type, init_state))
        hyp = cls(order)
        hyp.action_tree = node
        hyp.t = 0
        hyp.pending_nodes.append(node)
        hyp.unprocessed_nodes.append(node)
        return hyp


    def all_immediate_continous(self):
        pass

    def is_complete(self):
        return not self.pending_nodes
