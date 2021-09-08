def consolidate_tree(node):
    # only care explored children
    node.children = [x for x in node.children if x.verify_time != -1]
    # node.siblings = [x for x in node.siblings if x.verify_time != -1]
    for x in node.children:
        consolidate_tree(x)

def collect_nodes_and_edges(node):
    nodes = []
    tree_links = []
    visit_links = []
    _collect_nodes_and_edges(node, nodes, tree_links, visit_links, None, 0)

    return nodes, tree_links, visit_links

def _collect_nodes_and_edges(node, dot_nodes, tree_links, visit_links, parent, rank):
    # check whether to hide
    # to_show = True
    if parent is not None and len(node.children) == 1 and (parent.verify_time + 1) == node.verify_time:
        # parent_next_step = parent.next_step

        last = parent
        cursor = node
        child = node.children[0]
        while True:
            # print(last.verify_time, cursor.verify_time, child.verify_time)
            if cursor.verify_time == (last.verify_time + 1) and len(cursor.children) == 1 and child.verify_time == (cursor.verify_time + 1):
                last = cursor
                cursor = child
                if not child.children:
                    break
                child = cursor.children[0]
            else:
                break
        # visit_links.append
        # new_tree_link = '{} -> {} [style=dotted,color=red];'.format('node{}'.format(parent.verify_time), 'node{}'.format(cursor.verify_time))
        new_tree_link = '{} -> {};'.format('node{}'.format(parent.verify_time), 'node{}'.format(cursor.verify_time))

        new_visit_link = '{} -> {} [style=dashed; color=blue];'.format('node{}'.format(parent.verify_time), 'node{}'.format(cursor.verify_time))
        tree_links.append(new_tree_link)
        # visit_links.append(new_visit_link)
        node = cursor
    else:
        if parent is not None:
            # new_tree_link = '{} -> {} [style=dotted,color=red];'.format('node{}'.format(parent.verify_time), 'node{}'.format(node.verify_time))
            new_tree_link = '{} -> {};'.format('node{}'.format(parent.verify_time), 'node{}'.format(node.verify_time))
            new_visit_link = '{} -> {} [style=dashed; color=blue];'.format('node{}'.format(node.verify_time - 1), 'node{}'.format(node.verify_time))
            if parent.verify_time != node.verify_time - 1:
                visit_links.append(new_visit_link)
            tree_links.append(new_tree_link)

    node_name = 'node{}'.format(node.verify_time)
    debug_form = 'None' if node.partial_ast is None else node.partial_ast.short_debug_form()
    # if isinstance(node, Muta)
    if hasattr(node, 'search_score'):
        node_attr = '[label="T:{} S:{:.3f} NS:{:.3f}\n{}",color={}]'.format(node.verify_time, node.hypothesis.score, node.search_score, debug_form, 'black' if node.verify_result else 'red')
    else:
        node_attr = '[label="T:{} S:{:.3f}\n{}",color={}]'.format(node.verify_time, node.hypothesis.score, debug_form, 'black' if node.verify_result else 'red')
    new_node = '{} {};'.format(node_name, node_attr)
    dot_nodes.append((rank, new_node))
    for c in node.children:
        _collect_nodes_and_edges(c, dot_nodes, tree_links, visit_links, node, rank+1)

def make_dot_file(filename, explored):
    root = explored[0]
    # consolidate_tree(root)
    rank_node_pairs, tree_links, visit_links = collect_nodes_and_edges(root)

    max_ranks = max([x[0] for x in rank_node_pairs])
    # sort nodes by rank
    
    grouped_nodes = []
    for r in range(max_ranks + 1):
        grouped_nodes.append([x[1] for x in rank_node_pairs if x[0] == r])

    nodes = [x[1] for x in rank_node_pairs]
    with open(filename, 'w') as f:
        # f.write
        lines = ['digraph G{', 'node [shape=box];'] + nodes + tree_links + ['}']
        f.writelines([x + '\n' for x in lines])

        # lines = ['digraph G{', 'node [shape=box];']
        # for g_nodes in grouped_nodes:
            
        #     lines.append('{ rank=same;')
        #     lines.extend(g_nodes)
        #     lines.append('}')

        # lines.extend(tree_links)
        # lines.extend(visit_links)
        # lines.append('}')
        # f.writelines([x + '\n' for x in lines])
