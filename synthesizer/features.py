
# from synthesizer.synthesizer import SearchTreeNode
from components.dataset import *
from grammar.transition_system import PlaceHolderAction, ApplyRuleAction, GenTokenAction
from train_naive_searcher import IndexedFeature

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

def make_branch_info(fail_seq):

    final_hyp = fail_seq[-1].hypothesis
    node_seq, parents_ptr = linearize_action_tree(final_hyp)
    # exit()
    assert len(node_seq) == len(fail_seq)
    action_seq = []
    for i in range(len(fail_seq)):
        fn = fail_seq[i]
        action = fn.hypothesis.last_filled_node.action
        v_state = fn.parent.hypothesis.get_pending_node().action.v_state
        hole = fn.parent.hypothesis.get_pending_node().action
        score = fn.hypothesis.score - fn.parent.hypothesis.score
        action_seq.append(SynthAction(v_state, action, 'place', score, hole, node_seq[i]))
    return SynthBranch(action_seq, final_hyp, parents_ptr)


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

    # for a in branch.actions:
    #     print(a.score)
    features = []
    parent_ptrs = branch.parent_ptrs
    
    # len(parent_ptrs)
    failing_path = []
    error_idx = len(parent_ptrs) - 1
    while error_idx >= 0:
        failing_path.append(error_idx)
        error_idx = parent_ptrs[error_idx]
    failing_path.reverse()
        
    for i, a in enumerate(actions):
        features.append(extract_features_for_action(a, branch, i, failing_path))
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


def extract_features_for_action(a, branch, idx, failure_path):
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
    # feature.add('action_entropy')

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



def _make_np_feature(branch, features, indexer):
    new_feats = []
    for single_feat in features:
        val_feat = [.0] * len(indexer)
        for k, v in single_feat.data.items():
            if k in indexer:
                val_feat[indexer[k]] = v
        new_feats.append((val_feat))
    new_feats = np.array(new_feats)
    return new_feats

def predict_error(model, branch, features):
    indexed_feat = _make_np_feature(branch, features, model.feat_indexer)
    preditions = model.predict((indexed_feat, None))
    return preditions