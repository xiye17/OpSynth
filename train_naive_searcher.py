from common.config import *
from components.dataset import *
from grammar.grammar import Grammar

import time
from models import nn_utils

from torch import optim
from models.MetaLocator import *
from common.utils import *
from grammar.transition_system import ApplyRuleAction, GenTokenAction, PlaceHolderAction
from components.dataset import Dataset
import math
import random

def eval_np_locator_accuracy(predictions, examples, visualize=False):
    acc = 0
    for p, ex in zip(predictions, examples):
        # for l in zip(p, ex[1])
        # first 1
        # error idx
        predicted_idx = np.argmax(p)
        # print([x.label for x in ex[1].actions])
        gt_idx = [x.label for x in ex[1].actions].index('error')
        if gt_idx == predicted_idx:
            acc += 1
    acc = acc  / len(predictions)
    return acc


# def analyze_locator(predictions, examples, visualize=False):
#     acc = 0
#     for p, ex in zip(predictions, examples):
#         # for l in zip(p, ex[1])
#         # first 1
#         # error idx
#         predicted_idx = np.argmax(p)
#         # print([x.label for x in ex[1].actions])
#         gt_idx = [x.label for x in ex[1].actions].index('error')
#         p_list = p.tolist()
#         p_list.sort(reverse=True)
#         if gt_idx == predicted_idx:
#             print(ex[0].index, 'Match %.3f'%p[predicted_idx], '%.3f'%(p_list[0] - p_list[1]),  gt_idx, predicted_idx)
#         else:

#             print( ex[0].index, 'False %.3f'%p[predicted_idx],  '%.3f'%(p_list[0] - p_list[1]), gt_idx, predicted_idx)
#             print('Sorted', ' '.join(['%.2f'%x for x in p_list]))
#         print(' '.join(['%.2f'%x for x in p]))
    

#     acc = acc  / len(predictions)
#     return acc


def analyze_dismatch(predictions, examples, visualize=False):
    num_changed = 0
    num_effective = 0
    error_changes = []
    num_error = 0
    for p, ex in zip(predictions, examples):
        model_scores = np.exp(np.array([a.score for a in ex[1].actions]))
        predicted_certainty = 1 - p
        gt_idx = [x.label for x in ex[1].actions].index('error')

        # gap = np.abs()
        upweight_gap = 0.1
        predicted_idx = np.argmax(p)
        under_certained = np.logical_and(predicted_certainty - model_scores > upweight_gap, model_scores > 0.5)
        do_flag = False
        for i,u in enumerate(under_certained):
            if i > gt_idx:
                break
            if u and i != predicted_idx:
                do_flag = True
                num_changed += 1
                if i == gt_idx:
                    num_error += 1
                    error_changes.append('{:.2f}->{:.2f}'.format(model_scores[i], predicted_certainty[i]))
        if do_flag:
            num_effective += 1
    print(num_effective, num_changed, num_error)
    # return acc
    print(error_changes)


def analyze_locator(predictions, examples, visualize=False):
    acc = 0
    gt_pred_scores = []
    max_pred_scores = []
    # gt_model_scores = []
    for p, ex in zip(predictions, examples):
        predicted_idx = np.argmax(p)
        # print([x.label for x in ex[1].actions])
        gt_idx = [x.label for x in ex[1].actions].index('error')
        gt_pred_scores.append(p[gt_idx])
        max_pred_scores.append(p[predicted_idx])
    gt_pred_scores = np.array(gt_pred_scores)
    max_pred_scores = np.array(max_pred_scores)
    # print(gt_pred_scores.shape, max_pred_scores.shape)
    gap_scores = max_pred_scores - gt_pred_scores
    # print(gt_pred_scores)
    print(sum(gt_pred_scores < 0.05), sum(gt_pred_scores < 0.1), sum(gt_pred_scores < 0.15))
    gap_thres = 0.15
    true_thres = 0.05
    print(
        sum( np.logical_and(gt_pred_scores < 0.05, gap_scores > gap_thres)),
        sum( np.logical_and(gt_pred_scores < 0.1, gap_scores > gap_thres)),
        sum( np.logical_and(gt_pred_scores < 0.15, gap_scores > gap_thres))
    )
    # print(sorted(gap_scores.tolist()))
    # acc = acc  / len(predictions)
    

    # take effective

    sum_effective = 0
    sum_effective_err = 0
    effective_mark = 0.9
    num_affected = 0
    for p, ex in zip(predictions, examples):
        predicted_idx = np.argmax(p)
        # print([x.label for x in ex[1].actions])
        gt_idx = [x.label for x in ex[1].actions].index('error')

        # identified_as_error
        max_pred = p[predicted_idx]
        sured = np.logical_and(p < true_thres, (max_pred - p) > gap_thres)
        
        model_scores = np.exp(np.array([a.score for a in ex[1].actions]))
        
        effective_ones = np.logical_and(model_scores < effective_mark, sured)
        if sum(effective_ones):
            num_affected += 1
        if sured[gt_idx]:
            sum_effective_err += 1
        sum_effective += sum(effective_ones)
    print(sum_effective_err, sum_effective, num_affected)

    return acc



class IndexedFeature:
    def __init__(self):
        self.data = {}

    def add(self, k, v=1.0):
        self.data[k] = v
    
    def __getitem__(self, k):
        return self.data.get(k, 0.)

class FeatureVocab:
    def __init__(self):
        self.feat_to_id = {}
        self.id_to_feat = {}

    def __getitem__(self, word):
        return self.feat_to_id.get(word, -1)

    def __contains__(self, word):
        return word in self.feat_to_id

    def __len__(self):
        return len(self.feat_to_id)

    def size(self):
        return len(self.feat_to_id)

    def get_word(self, wid):
        return self.id_to_feat[wid]

    def add(self, word):
        if word not in self:
            wid = self.feat_to_id[word] = len(self)
            self.id_to_feat[wid] = word
            return wid

# version1: look at the first failed branch only
# example should contain (data_example, first fail branch)
def path_to_dataset(dataset, path, top_k=5):
    examples = []
    for i, (data, p) in enumerate(zip(dataset, path)):
        data.index = i
        for b in p.branches[:top_k]:
            examples.append((data, b, extract_features_for_branch(b)))
    return examples

# version1: look at the first failed branch only
# example should contain (data_example, first fail branch)
def path_to_dataset_randomly(dataset, path, top_k=10):
    examples = []
    random.seed(666)
    for i, (data, p) in enumerate(zip(dataset, path)):
        if not p.branches:
            continue
        data.index = i
        # for b in p.branches[:top_k]:
        branches = p.branches
        examples.append((data, branches[0], extract_features_for_branch(branches[0])))
        num_left = min(top_k, len(branches) - 1)
        if num_left:
            for b in random.sample(branches[1:], num_left):
                examples.append((data, b, extract_features_for_branch(b)))
    return examples

def build_feature_bank(dataset):
    feature_vocab = FeatureVocab()
    for _, _, features in dataset:
        for feat in features:
            for fname in feat.data:
                feature_vocab.add(fname)

    return feature_vocab


def sorted_str_repr_for_candidates(hole):
    scores = hole.candidate_scores.tolist()
    dsl_type = hole.type
    if dsl_type.is_composite_type():
        candidates = [x.constructor.name for x in hole.candidates]
    else:
        vocab = hole.candidates
        candidates = [ vocab.id_to_word[x] for x in range(len(vocab))]
    sorted_pairs = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [x[1] for x in sorted_pairs]

def get_hole_aux_info(branch):
    hole_aux_infos = []
    for a in branch.actions:
        if a.hole.type.name in ['str', 'int', 'tok']:
            hole_aux_infos.append(None)
            continue
        # print()
        candidate_repr = sorted_str_repr_for_candidates(a.hole)
        hole_aux_infos.append(candidate_repr)
    return hole_aux_infos

def extract_features_for_branch(branch):
    actions = branch.actions

    features = []
    parent_ptrs = branch.parent_ptrs
    
    # len(parent_ptrs)
    failing_path = []
    error_idx = len(parent_ptrs) - 1
    while error_idx >= 0:
        failing_path.append(error_idx)
        error_idx = parent_ptrs[error_idx]
    failing_path.reverse()

    hole_aux_infos = get_hole_aux_info(branch)
        
    for i, a in enumerate(actions):
        features.append(extract_features_for_action(a, branch, i, failing_path, hole_aux_infos))
    return features

def str_repr_for_action(ac):
    if isinstance(ac, ApplyRuleAction):
        return ac.production.constructor.name
    elif isinstance(ac, GenTokenAction):
        type_name = ac.type.name
        if type_name in ['cc', 'csymbl']:
            return ac.token
        else:
            return type_name
    elif isinstance(ac, PlaceHolderAction):
        return 'hole'


def extract_features_for_action(a, branch, idx, failure_path, hole_rank_infos):
    feature = IndexedFeature()
    node_info = branch.actions[idx].node
    # hole_info = branch.actions[idx].hole
    # info of the whole tree
    # root action
    root_action = str_repr_for_action(branch.actions[0].action)
    feature.add('root_action:'+root_action)
    
    # info of this action
    this_action_repr = str_repr_for_action(a.action)
    feature.add('this_action:'+this_action_repr)
    feature.add('this_depth_cat:' + str(node_info.depth))
    feature.add('this_depth_val:', str(node_info.depth/5.0))
    # if not node_info.children:
    #     feature.add('is_a_leaf_node')

    # action sequence, parent grand parent, children, siblings
    parent = node_info.parent
    feature.add('parent_action:' + ('none' if parent is None else str_repr_for_action(parent.action) ))
    # grand parent
    if parent is None:
        feature.add('gradparent_action:' + 'none')
    else:
        grad_parent = parent.parent
        feature.add('gradparent_action:' + ('none' if grad_parent is None else str_repr_for_action(grad_parent.action) ))

    
    if node_info.fields:
        for child_id, child in enumerate(node_info.fields[:2]):
            feature.add('child'+str(child_id) + ':' + str_repr_for_action(child.action))
        if node_info.fields[0].fields:
            feature.add('gradchild0:' + str_repr_for_action(node_info.fields[0].fields[0].action))
        else:
            feature.add('gradchild0:none')
    else:
        feature.add('children:none')
    
    # scores related
    # prob = math.exp(a.score)
    prob = (a.score)
    feature.add('choice_prob:', prob)
    # candidate_scores = a.hole.candidate_scores
    # entropy = - torch.sum(torch.exp(candidate_scores) * candidate_scores).item()
    # feature.add('action_entropy', entropy)


    # time in the id
    # feature.add('visited_time:', idx/10.0)
    feature.add('visited_val:', idx/len(branch.actions))

    # on failing path
    if idx in failure_path:
        feature.add('on_failing_path')

    # window for next
    # neb_n_choice
    # feature.add('')
    # rank_info = hole_rank_infos[idx]
    # if rank_info:
    #     feature.add('topac:' + rank_info[0])
    #     feature.add('secac:' + rank_info[1])
    #     feature.add('thdac:' + rank_info[2])
    #     # print(this_action_repr, rank_info)
    #     if this_rank + 1 < len(rank_info):
    #         feature.add('next_candidate:'+rank_info[this_rank + 1])
    #     else:
    #         feature.add('next_candidate:none')
    # if rank_info:
    #     this_rank = rank_info.index(this_action_repr)
    #     feature.add('this_rank_val:', this_rank/len(rank_info))

    return feature


# def index_dataset(indexer)
def train_naive_searcher(args):
    train_data = easy_pickle_read(args.train_file)
    train_path = easy_pickle_read(args.train_path_file)

    dev_data = easy_pickle_read(args.dev_file)
    dev_path = easy_pickle_read(args.dev_path_file)

    train_set = path_to_dataset_randomly(train_data, train_path, top_k=20)
    dev_set = path_to_dataset(dev_data, dev_path,top_k=1)

    feature_indexer = build_feature_bank(train_set)

    model = RandomForestLocator(feature_indexer)
    train_set_indexed = model.pre_index_examples(train_set)
    dev_set_indexed = model.pre_index_examples(dev_set)

    model.fit(train_set_indexed)
    model.interpretablity()
    # train_predictions = [model.predict(ex) for ex in train_set_indexed]
    # print('Train', eval_np_locator_accuracy(train_predictions, train_set))
    easy_pickle_dump(model, 'rf_locator.pkl')
    dev_predictions = [model.predict(ex) for ex in dev_set_indexed]
    print('Dev', eval_np_locator_accuracy(dev_predictions, dev_set))

    # analyze_locator(dev_predictions, dev_set)
    # analyze_dismatch(dev_predictions, dev_set)

def train_naivenn_searcher(args):
    train_data = easy_pickle_read(args.train_file)
    train_path = easy_pickle_read(args.train_path_file)

    dev_data = easy_pickle_read(args.dev_file)
    dev_path = easy_pickle_read(args.dev_path_file)

    train_set = path_to_dataset(train_data, train_path)
    dev_set = path_to_dataset(dev_data, dev_path)

    feature_indexer = build_feature_bank(train_set)


    model = ScratchLocator(args, parser=parser)
    # model = SeqToSeqLocator(args, parser=parser)
    # model = SpecificLocator(args, parser=parser)

    train_set_indexed = model.pre_index_examples(train_set)
    dev_set_indexed = model.pre_index_examples(dev_set)

    optimizer = optim.Adam(model.parameters(), lr=0.003)
    args.max_epoch = 500
    for epoch in range(1, args.max_epoch + 1):

        model.train()
        epoch_begin = time.time()
        optimizer.zero_grad()
        loss = model.score(train_set_indexed)
        loss = torch.mean(loss)
        loss_val = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()
        print('[epoch {}] train loss {:.3f}, epoch time {:.0f}'.format(epoch, loss_val , time.time() - epoch_begin) )
        model.eval()
        with torch.no_grad():
            train_predictions = [model.predict(ex) for ex in train_set_indexed]
        print('Train', eval_locator_accuracy(train_predictions, train_set))
        with torch.no_grad():
            dev_predictions = [model.predict(ex) for ex in dev_set_indexed]
        print('Dev', eval_locator_accuracy(dev_predictions, dev_set))

if __name__ == "__main__":
    args = parse_args('train_searcher')
    train_naive_searcher(args)
    # train_naivenn_searcher(args)
